"""Declining a folded trijunction observation: identity at zero, and the accounting.

These pin `foambryo.wedge_refusal`. The scientific regression they exist to catch is the one that
makes a refusal policy dangerous rather than merely wrong: an estimator that silently returns a
number for a quantity its surviving rows no longer constrain. Every test below is therefore either
an *identity* (nothing refused, nothing moves — bitwise) or an *accounting* (what was refused, what
survived, and what came back `nan`), and none of them looks at an accuracy statistic.

The two structural facts the policies' definitions imply are asserted here rather than assumed:

* the per-edge estimator's observation unit is a single edge, so `ExcludeEdgeFromLineMean` and
  `DropLine` are the same intervention there and must agree **exactly**;
* for a line-averaged estimator `ExcludeEdgeFromLineMean` and `DownweightLine` give a surviving
  line the *same* row weight, so those two policies differ only in the angle a line reports.

Most tests drive the machinery with an **explicit** refused-edge set rather than waiting for a
folded mesh: the policies' plumbing is then exercised at any refusal size on the repository's own
golden-master mesh, and the folded-mesh path is covered separately by the reference meshes that
actually carry reflex edges.
"""

import warnings

import numpy as np
import pytest

from foambryo import (
    TensionComputationMethod,
    dcel_mesh_from_multimaterial_mesh,
    infer_tensions,
    infer_tensions_by_junction_component,
)
from foambryo.junction_graph import JunctionGraphGaugeWarning
from foambryo.validation import mesh_io
from foambryo.wedge_refusal import (
    NO_REFUSAL,
    WedgeRefusal,
    WedgeRefusalPolicy,
    line_averaged_angles,
    per_edge_rows,
    reflex_trijunction_edges,
    refusal_report,
)

from .conftest import DATASET_DIR, require_substitute_mesh

#: Every method whose rows are built from measured wedge angles, and which the selector reaches.
ANGLE_METHODS = (
    TensionComputationMethod.YoungDupre,
    TensionComputationMethod.YoungDupreLocal,
    TensionComputationMethod.ProjectionYoungDupre,
    TensionComputationMethod.Lami,
    TensionComputationMethod.InvLami,
    TensionComputationMethod.LogLami,
    TensionComputationMethod.Cotan,
    TensionComputationMethod.InvCotan,
)

#: Every method that assembles an operator instead, and must therefore refuse a refusal.
NON_ANGLE_METHODS = (
    TensionComputationMethod.Equilibrium,
    TensionComputationMethod.Variational,
    TensionComputationMethod.ForceBalance,
    TensionComputationMethod.ForceBalanceSoftRow,
    TensionComputationMethod.ForceBalanceMAP,
)

POLICIES = (
    WedgeRefusalPolicy.ExcludeEdgeFromLineMean,
    WedgeRefusalPolicy.DropLine,
    WedgeRefusalPolicy.DownweightLine,
)

#: The reference meshes this repository keeps as a sibling that do carry folded wedges, with the
#: count pinned rather than asserted to be zero: zero is false, and a *rise* here would mean the
#: reference geometry itself had moved.
REFERENCE_MESHES_WITH_FOLDS = {"004": 2, "020": 2, "045": 1}


def _load(path):
    v, f, r = mesh_io.read_rec(path)
    return dcel_mesh_from_multimaterial_mesh(v, f, r)


@pytest.fixture(scope="module")
def clean_mesh():
    """The repository's golden-master mesh, which carries no folded wedge."""
    return _load(require_substitute_mesh())


def _reference_mesh(case: str):
    path = DATASET_DIR / f"{case}_mesh.rec"
    if not path.exists():
        pytest.skip(f"{path} not found (sibling benchmarking-dataset/ missing)")
    return _load(path)


def _tensions(mesh, method, refusal, reading="oriented_chord"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return infer_tensions(mesh, 1.0, method, angle_reading=reading, refusal=refusal)


def _same(left: dict, right: dict) -> bool:
    """Bitwise equality of two tension dictionaries, `nan` counting as equal to `nan`."""
    if set(left) != set(right):
        return False
    return all(
        left[key] == right[key] or (np.isnan(left[key]) and np.isnan(right[key]))
        for key in left
    )


def _some_edges(mesh, fraction: float = 0.2) -> frozenset:
    """A deterministic, reproducible slice of this mesh's trijunctional edges."""
    from foambryo.geometry import compute_trijunction_edges

    edges = compute_trijunction_edges(mesh, reading="oriented_chord")
    step = max(1, round(1.0 / fraction))
    return frozenset(edge.vertices for edge in edges[::step])


# --------------------------------------------------------------------------------------------
# T1 -- identity when nothing is refused
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ANGLE_METHODS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("reading", ["chord", "oriented_chord"])
def test_a_policy_with_nothing_to_refuse_is_bitwise_the_shipped_estimator(clean_mesh, method, policy, reading):
    """On a mesh with no folded wedge, every policy reproduces the shipped answer bitwise.

    This is the safeguard the whole selector rests on: a policy is only ever allowed to change a
    number where the mesh is actually folded. `YoungDupreLocal` refuses the quadric readings, so
    only the two chord readings are exercised -- which are the two the refusal is defined in.
    """
    assert not reflex_trijunction_edges(clean_mesh)
    base = _tensions(clean_mesh, method, NO_REFUSAL, reading)
    refused = _tensions(clean_mesh, method, WedgeRefusal(policy=policy), reading)
    assert _same(base, refused)


def test_the_default_refusal_is_no_refusal(clean_mesh):
    """Not passing a refusal and passing the default one are the same call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implicit = infer_tensions(clean_mesh, 1.0, TensionComputationMethod.YoungDupre)
    explicit = _tensions(clean_mesh, TensionComputationMethod.YoungDupre, NO_REFUSAL, "chord")
    assert _same(implicit, explicit)
    assert NO_REFUSAL.refuses_nothing


@pytest.mark.parametrize(("case", "expected"), sorted(REFERENCE_MESHES_WITH_FOLDS.items()))
def test_the_reference_meshes_that_do_fold_keep_their_pinned_count(case, expected):
    """The three reference meshes carrying folded wedges, pinned by count.

    Pinned rather than asserted to be zero, because zero is false: these are the deliberate
    non-trivial fixture, and a *rise* in the count would mean the reference geometry had moved.
    """
    mesh = _reference_mesh(case)
    assert len(reflex_trijunction_edges(mesh)) == expected


@pytest.mark.parametrize("case", sorted(REFERENCE_MESHES_WITH_FOLDS))
def test_a_folded_reference_mesh_moves_only_where_it_folds(case):
    """On the folded reference meshes a policy does change the answer, and is still well posed.

    The complement of the identity test: it would be possible to pass every check above with a
    selector that did nothing at all.
    """
    mesh = _reference_mesh(case)
    base = _tensions(mesh, TensionComputationMethod.YoungDupre, NO_REFUSAL)
    dropped = _tensions(mesh, TensionComputationMethod.YoungDupre, WedgeRefusal(policy=WedgeRefusalPolicy.DropLine))
    assert not _same(base, dropped)
    assert np.isfinite([value for value in dropped.values() if not np.isnan(value)]).all()


# --------------------------------------------------------------------------------------------
# T2, T3 -- the two structural identities the definitions imply
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_exclude_and_drop_coincide_for_the_per_edge_estimator(clean_mesh, fraction):
    """The per-edge estimator's unit is one edge, so excluding it and dropping it are one thing.

    Exact, not approximate: both policies must assemble the identical row set with the identical
    weights, so the two solves agree bitwise at every refusal size.
    """
    edges = _some_edges(clean_mesh, fraction)
    exclude = _tensions(
        clean_mesh,
        TensionComputationMethod.YoungDupreLocal,
        WedgeRefusal(policy=WedgeRefusalPolicy.ExcludeEdgeFromLineMean, edges=edges),
    )
    drop = _tensions(
        clean_mesh,
        TensionComputationMethod.YoungDupreLocal,
        WedgeRefusal(policy=WedgeRefusalPolicy.DropLine, edges=edges),
    )
    assert _same(exclude, drop)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_exclude_and_downweight_give_a_line_the_same_row_weight(clean_mesh, fraction):
    """A surviving line is weighted identically under `E` and `W`; only its angle differs.

    `E`'s weight is the line's surviving arclength; `W`'s is its full arclength times the
    surviving fraction. Equal in exact arithmetic, and equal here to floating-point accumulation
    order -- the two sum the same lengths in a different order, so the tolerance is on the
    relative difference and not on the bits.
    """
    edges = _some_edges(clean_mesh, fraction)
    exclude = line_averaged_angles(
        clean_mesh,
        reading="oriented_chord",
        refusal=WedgeRefusal(policy=WedgeRefusalPolicy.ExcludeEdgeFromLineMean, edges=edges),
    )
    downweight = line_averaged_angles(
        clean_mesh,
        reading="oriented_chord",
        refusal=WedgeRefusal(policy=WedgeRefusalPolicy.DownweightLine, edges=edges),
    )
    assert set(exclude.dict_length) == set(downweight.dict_length)
    for key, weight in downweight.dict_length.items():
        assert exclude.dict_length[key] == pytest.approx(weight, rel=1e-12)
    # ... and the angles are what separates them, wherever a line actually lost an edge.
    assert any(exclude.dict_rad[key] != downweight.dict_rad[key] for key in downweight.dict_rad)


def test_drop_line_removes_strictly_more_lines_than_exclude(clean_mesh):
    """`D` is the strictly harsher line rule, and `W` removes no line at all."""
    edges = _some_edges(clean_mesh, 0.25)
    kept = {}
    for policy in POLICIES:
        angles = line_averaged_angles(
            clean_mesh,
            reading="oriented_chord",
            refusal=WedgeRefusal(policy=policy, edges=edges),
        )
        kept[policy] = {tuple(sorted(key)) for key in angles.dict_rad}
    assert kept[WedgeRefusalPolicy.DropLine] < kept[WedgeRefusalPolicy.ExcludeEdgeFromLineMean]
    assert kept[WedgeRefusalPolicy.DownweightLine] == kept[WedgeRefusalPolicy.ExcludeEdgeFromLineMean]


# --------------------------------------------------------------------------------------------
# T4 -- the accounting
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("per_edge", [False, True])
def test_the_refused_and_used_counts_sum_to_the_units_offered(clean_mesh, policy, per_edge):
    """Units in, units refused, units used -- the three must close, in either unit."""
    edges = _some_edges(clean_mesh, 0.25)
    report = refusal_report(clean_mesh, WedgeRefusal(policy=policy, edges=edges), per_edge_rows=per_edge)
    assert report.n_units_refused + report.n_units_used == report.n_units
    assert report.n_edges_refused == len(edges)
    assert 0 <= report.length_refused <= report.length_total
    assert report.n_lines_without_row + len(report.surviving_lines) == report.n_lines


@pytest.mark.parametrize("policy", POLICIES)
def test_the_report_agrees_with_the_rows_the_estimators_actually_assemble(clean_mesh, policy):
    """The counts are the rows, not a parallel estimate of them."""
    from foambryo.geometry import compute_trijunction_edges

    edges = _some_edges(clean_mesh, 0.25)
    refusal = WedgeRefusal(policy=policy, edges=edges)

    line_report = refusal_report(clean_mesh, refusal)
    angles = line_averaged_angles(clean_mesh, reading="oriented_chord", refusal=refusal)
    assert line_report.n_units_used == len({tuple(sorted(key)) for key in angles.dict_rad})
    assert set(line_report.surviving_lines) == {tuple(sorted(key)) for key in angles.dict_rad}

    edge_report = refusal_report(clean_mesh, refusal, per_edge_rows=True)
    rows = per_edge_rows(clean_mesh, compute_trijunction_edges(clean_mesh, reading="oriented_chord"), refusal)
    assert edge_report.n_units_used == len(rows.keep)
    assert edge_report.n_units_refused == rows.n_refused


@pytest.mark.parametrize("policy", POLICIES)
def test_an_interface_left_with_no_surviving_line_is_reported_undetermined(clean_mesh, policy):
    """Refusing every line that touches one interface makes that interface `nan`, not a number.

    This is the whole point of rebuilding the gauge partition over the surviving lines: ordinary
    least squares would have returned a minimum-norm number for an interface nothing constrains.
    """
    from foambryo.geometry import compute_trijunction_edges

    all_edges = compute_trijunction_edges(clean_mesh, reading="oriented_chord")
    target = min({key[1] for edge in all_edges for key in edge.keys})
    doomed = frozenset(
        edge.vertices for edge in all_edges if any(target in (key[0], key[1], key[2]) for key in edge.keys)
    )
    assert doomed

    refusal = WedgeRefusal(policy=policy, edges=doomed)
    solution = infer_tensions_by_junction_component(
        clean_mesh,
        1.0,
        TensionComputationMethod.YoungDupre,
        refusal=refusal,
    )
    touching = [key for key in solution.dict_tensions if target in key]
    assert touching
    assert all(np.isnan(solution.dict_tensions[key]) for key in touching)
    assert set(touching) <= set(solution.undetermined)
    assert all(solution.component_of[key] is None for key in touching)

    # The reporting entry point above stays silent by design -- it hands the caller the structure
    # the warning would have described. The flat one, which drops that structure, must warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flat = infer_tensions(clean_mesh, 1.0, TensionComputationMethod.YoungDupre, refusal=refusal)
    assert any(issubclass(record.category, JunctionGraphGaugeWarning) for record in caught)
    assert all(np.isnan(flat[key]) for key in touching)


@pytest.mark.parametrize("method", ANGLE_METHODS)
@pytest.mark.parametrize("policy", POLICIES)
def test_refusing_everything_returns_an_empty_answer_rather_than_raising(clean_mesh, method, policy):
    """A policy that leaves no observation at all answers `nan` everywhere, and does not raise.

    An empty answer is a result the caller can count; an exception is one they cannot compare
    against the policies that did produce an answer.
    """
    from foambryo.geometry import compute_trijunction_edges

    everything = frozenset(edge.vertices for edge in compute_trijunction_edges(clean_mesh, reading="oriented_chord"))
    tensions = _tensions(clean_mesh, method, WedgeRefusal(policy=policy, edges=everything))
    assert tensions
    assert all(np.isnan(value) for value in tensions.values())


# --------------------------------------------------------------------------------------------
# T5 -- every method's call path, checked individually
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", NON_ANGLE_METHODS)
def test_a_method_that_reads_no_wedge_angle_refuses_a_policy(clean_mesh, method):
    """Silently ignoring a policy would hand the caller a result they believe refused and is not."""
    with pytest.raises(ValueError, match="does not build its rows from measured trijunction wedge angles"):
        infer_tensions(clean_mesh, 1.0, method, refusal=WedgeRefusal(policy=WedgeRefusalPolicy.DropLine))


@pytest.mark.parametrize("method", NON_ANGLE_METHODS)
def test_a_method_that_reads_no_wedge_angle_still_accepts_the_default(clean_mesh, method):
    """The refusal of §T5 is on the policy, not on the parameter: the default must go through."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explicit = infer_tensions(clean_mesh, 1.0, method, refusal=NO_REFUSAL)
        implicit = infer_tensions(clean_mesh, 1.0, method)
    assert _same(explicit, implicit)


@pytest.mark.parametrize("method", ANGLE_METHODS)
def test_every_angle_method_honours_the_selector(clean_mesh, method):
    """Each angle-reading method's own call path is reached, and each one actually changes."""
    edges = _some_edges(clean_mesh, 0.25)
    base = _tensions(clean_mesh, method, NO_REFUSAL)
    refused = _tensions(clean_mesh, method, WedgeRefusal(policy=WedgeRefusalPolicy.DropLine, edges=edges))
    assert not _same(base, refused)


# --------------------------------------------------------------------------------------------
# T6 -- the refusal reads nothing but the mesh
# --------------------------------------------------------------------------------------------


def test_the_refusal_reads_nothing_but_the_mesh(clean_mesh):
    """No reference geometry, target angle or expected value can reach the decision.

    Asserted on the signature, because that is the check that keeps holding when the body
    changes, and then operationally on a mesh rebuilt from bare arrays with no file behind it.
    """
    import inspect

    parameters = list(inspect.signature(reflex_trijunction_edges).parameters)
    assert parameters == ["mesh", "tolerance_deg"]

    rebuilt = dcel_mesh_from_multimaterial_mesh(
        np.array(clean_mesh.v, dtype=np.float64),
        np.array(clean_mesh.f[:, :3], dtype=np.int64),
        np.array(clean_mesh.f[:, 3:5], dtype=np.int64),
    )
    assert reflex_trijunction_edges(rebuilt) == reflex_trijunction_edges(clean_mesh)


def test_an_explicit_edge_set_is_normalised_and_never_reinterpreted():
    """An explicit set is a set of edge keys; nothing else about it is consulted."""
    refusal = WedgeRefusal(policy=WedgeRefusalPolicy.DropLine, edges=[(7, 3), (3, 7), (1, 2)])
    assert refusal.edges == frozenset({(3, 7), (1, 2)})
