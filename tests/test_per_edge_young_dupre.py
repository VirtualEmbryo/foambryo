"""The per-edge Young-Dupre work: pin the per-edge method, the `unique` resolution, and the edge-multiplicity question.

Three things are pinned here and they are of different kinds, so they are labelled:

1. **Correctness identities** — the per-edge view and the averaged view must agree wherever
   they are mathematically obliged to. These would catch a real regression.
2. **The `unique` resolution** (`geometry.compute_angles_tri`) — `unique=False` must stay
   bit-identical to the historical behaviour every production call site uses, and `unique=True`
   must now de-duplicate for real.
3. **The negative result** — `YoungDupreLocal` is *worse* than `YoungDupre` on real meshes, and
   that is asserted as a standing fixture rather than left as prose in a log. If a future change
   makes per-edge competitive, this test fails and somebody has to revisit the measurement and
   decide, which is the intent.
"""

from __future__ import annotations

import numpy as np
import pytest

from foambryo_benchmarks import metrics as m
from foambryo_benchmarks import phantoms as ph
from foambryo_benchmarks.run_per_edge_young_dupre import constructed_open_boundary_case, multiplicity_audit
from foambryo import dcel_mesh_from_file, infer_tensions
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri, compute_trijunction_edges
from foambryo.junction_graph import solve_with_exact_gauge
from foambryo.per_edge_tension_inference import (
    DEFAULT_HUBER_DELTA,
    PerEdgeWeighting,
    _assemble_per_edge_blocks,
    build_matrix_tension_local_yd,
    infer_tensions_young_dupre_local,
)
from foambryo.tension_inference import TensionComputationMethod, _infer_tension_equilibrium
from tests.conftest import require_substitute_mesh


@pytest.fixture(scope="module")
def mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


@pytest.fixture(scope="module")
def doublet():
    v, f, regions = ph.doublet_mesh(1.5, n_rings=16, n_phi=32)
    return DcelData(v, f, regions)


# --------------------------------------------------------------------------------------------
# 1. Correctness identities
# --------------------------------------------------------------------------------------------


def test_per_edge_angles_average_to_compute_angles_tri(mesh):
    """The per-edge view must reduce to the averaged view. Same convention, one helper."""
    rad, _, _ = compute_angles_tri(mesh, unique=True)
    per_key: dict[tuple[int, int, int], list[float]] = {}
    for edge in compute_trijunction_edges(mesh):
        for key, angle in zip(edge.keys, edge.angles, strict=True):
            per_key.setdefault(key, []).append(angle)

    assert set(per_key) == set(rad)
    for key, angles in per_key.items():
        assert np.mean(angles) == pytest.approx(rad[key], abs=1e-12)


def test_per_edge_lengths_sum_to_the_unique_accumulator(mesh):
    """`unique=True`'s accumulated length is exactly the sum of the per-edge lengths."""
    _, _, lengths = compute_angles_tri(mesh, unique=True)
    per_key: dict[tuple[int, int, int], float] = {}
    for edge in compute_trijunction_edges(mesh):
        for key in edge.keys:
            per_key[key] = per_key.get(key, 0.0) + edge.length
    for key, length in per_key.items():
        assert length == pytest.approx(lengths[key], rel=1e-12)


def test_every_trijunctional_edge_is_visited_exactly_three_times(mesh):
    """The edge-multiplicity question, interior half: the visit multiplicity is uniform on a closed mesh."""
    assert {len(edge.half_edge_keys) for edge in compute_trijunction_edges(mesh)} == {3}


def test_young_dupre_local_equals_young_dupre_when_the_angle_is_constant(doublet):
    """The reduction identity: with a constant angle along the line, per-edge == averaged.

    The doublet's single circular triple line has a constant wedge angle by symmetry, so the
    Jensen gap of `cos` vanishes and the two methods must coincide. This is what makes the
    real-mesh gap attributable to along-line angle variation and nothing else.
    """
    averaged = infer_tensions(doublet, 1, TensionComputationMethod.YoungDupre)
    local = infer_tensions(doublet, 1, TensionComputationMethod.YoungDupreLocal)
    assert set(averaged) == set(local)
    for key in averaged:
        assert local[key] == pytest.approx(averaged[key], rel=1e-9)


def test_line_preserving_weights_reproduce_young_dupre_with_averaged_angles(mesh):
    """The control that proves the row split is neutral: only the *reweighting* costs anything.

    Give every edge its line's averaged angles and weights whose squares sum, per line, to
    `YoungDupre`'s `(l_k / L)^2`. The per-edge system must then return `YoungDupre`'s tensions
    to machine precision — so any measured difference between the two methods is the angles,
    not the row count.
    """
    dict_areas = mesh.compute_areas_interfaces()
    keys = [tuple(int(label) for label in key) for key in sorted(dict_areas)]
    edges = compute_trijunction_edges(mesh)
    blocks, columns, regions, lengths, _ = _assemble_per_edge_blocks(edges, dict_areas)
    _lines, line_index = np.unique(regions, axis=0, return_inverse=True)
    line_index = line_index.ravel()
    line_lengths = np.bincount(line_index, weights=lengths)
    line_counts = np.bincount(line_index)

    rad, _, _ = compute_angles_tri(mesh, unique=True)
    averaged_blocks = np.empty_like(blocks)
    for edge_i in range(len(blocks)):
        a, b, c = (int(label) for label in regions[edge_i])
        angle_a = rad[(min(b, c), a, max(b, c))]
        angle_b = rad[(min(a, c), b, max(a, c))]
        angle_c = rad[(min(a, b), c, max(a, b))]
        averaged_blocks[edge_i] = [
            [1.0, np.cos(angle_b), np.cos(angle_a)],
            [np.cos(angle_b), 1.0, np.cos(angle_c)],
            [np.cos(angle_a), np.cos(angle_c), 1.0],
        ]

    total_length = lengths.sum()
    weights = line_lengths[line_index] / total_length / np.sqrt(line_counts[line_index])
    matrix_m, vector_b = build_matrix_tension_local_yd(averaged_blocks, columns, weights, len(keys), 1.0)
    # The same exact-gauge solve both methods use (`foambryo.junction_graph.solve_with_exact_gauge`).
    tensions, *_ = solve_with_exact_gauge(matrix_m[:-1], vector_b[:-1], float(vector_b[-1]))

    reference = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre)
    for key, value in zip(keys, tensions, strict=True):
        assert value == pytest.approx(reference[key], abs=1e-10)


def test_residual_field_is_aligned_and_finite(mesh):
    """The per-edge residual field must be one entry per counted edge, and renderable."""
    result = infer_tensions_young_dupre_local(mesh)
    field = result.residual_field
    n = result.n_edges
    assert field.vertices.shape == (n, 2)
    assert field.midpoints.shape == (n, 3)
    assert field.residual_vectors.shape == (n, 3)
    assert field.residual_norms.shape == (n,)
    assert field.regions.shape == (n, 3)
    assert np.all(np.isfinite(field.residual_norms))
    assert np.all(field.lengths > 0)
    assert result.n_rows == 3 * n + 1
    # Every midpoint must lie between its two vertices.
    expected = 0.5 * (mesh.v[field.vertices[:, 0]] + mesh.v[field.vertices[:, 1]])
    assert np.allclose(field.midpoints, expected)


def test_robust_weighting_only_downweights(mesh):
    """IRLS multipliers are in (0, 1]: robust weighting may discount an edge, never amplify it."""
    result = infer_tensions_young_dupre_local(mesh, huber_delta=DEFAULT_HUBER_DELTA)
    assert np.all(result.residual_field.robust_weights > 0)
    assert np.all(result.residual_field.robust_weights <= 1.0 + 1e-12)
    assert result.irls_iterations > 0


def test_young_dupre_local_is_registered_and_runs_like_the_others(mesh):
    """`YoungDupreLocal` must behave as a first-class enum member, not a bolt-on."""
    tensions = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupreLocal)
    reference = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre)
    assert set(tensions) == set(reference)
    assert np.mean(list(tensions.values())) == pytest.approx(1.0, abs=1e-2)


# --------------------------------------------------------------------------------------------
# 2. The `unique` resolution
# --------------------------------------------------------------------------------------------


def test_unique_false_is_the_historical_six_fold_accumulation(mesh):
    """`unique=False` — every production call site — must not move. Angles equal, lengths 6x.

    Historically `unique=True` and `unique=False` differed only by a uniform factor 2 and the
    parameter was dead. The per-edge Young-Dupre work gave `unique=True` a real meaning; `unique=False` keeps
    the old accumulation exactly, which is what keeps every inferred number unchanged.
    """
    rad_unique, _, lengths_unique = compute_angles_tri(mesh, unique=True)
    rad_all, _, lengths_all = compute_angles_tri(mesh, unique=False)

    assert set(rad_unique) == set(rad_all)
    for key in rad_unique:
        assert rad_unique[key] == pytest.approx(rad_all[key], rel=1e-12)
        assert lengths_all[key] == pytest.approx(6 * lengths_unique[key], rel=1e-12)


def test_unique_true_makes_total_length_mesh_the_actual_total_length(mesh):
    """`sum(dict_length) / 3` — what every builder calls `total_length_mesh` — is now true.

    Historically, `total_length_mesh` was not the total length of the mesh. Under `unique=True` it
    is, exactly, because each geometric edge is counted once per its three wedge keys.
    """
    _, _, lengths_unique = compute_angles_tri(mesh, unique=True)
    _, _, lengths_all = compute_angles_tri(mesh, unique=False)
    true_total = sum(edge.length for edge in compute_trijunction_edges(mesh))

    assert sum(lengths_unique.values()) / 3 == pytest.approx(true_total, rel=1e-12)
    # And the historical accumulator is exactly 6x that, which is the over-count the ratio cancels.
    assert sum(lengths_all.values()) / 3 == pytest.approx(6 * true_total, rel=1e-12)


# --------------------------------------------------------------------------------------------
# 3. The edge-multiplicity question, and the recorded negative results
# --------------------------------------------------------------------------------------------


def test_to_verify_3_multiplicity_is_uniform_but_open_boundaries_drop_edges():
    """The edge-multiplicity question, resolved: the multiplicity does not change — the *edge set* does.

    A triple line running off an open mesh boundary keeps three incident half-edges on every
    edge that still has three films, so the visit multiplicity is *not* the biasing mechanism a
    code review had conjectured. The bias is real but arrives differently: an edge that has lost one of its
    three films has only two incident faces, fails `len(twin) > 1`, and is dropped entirely — so
    the line's junction-length weight is under-stated in proportion to the lost fraction.
    """
    case = constructed_open_boundary_case(n_segments=6, intact_segments=4)

    assert case["intact"]["visit_multiplicities"] == [3]
    assert case["truncated"]["visit_multiplicities"] == [3]
    assert case["intact"]["n_counted_trijunctional_edges"] == 6
    assert case["truncated"]["n_counted_trijunctional_edges"] == 4
    assert case["weight_bias_factor"] == pytest.approx(4 / 6, rel=1e-9)


def test_dataset_meshes_are_closed_so_the_boundary_bias_cannot_fire(mesh):
    """On a real mesh there is no edge with fewer than two incident faces, so no line is clipped."""
    audit = multiplicity_audit(mesh)
    assert audit["n_mesh_boundary_edges"] == 0
    assert audit["visit_multiplicities"] == [3]
    # Only quadrijunctional edges are dropped, and they are dropped on purpose (Plateau).
    assert audit["n_counted_trijunctional_edges"] + audit["n_dropped_quadrijunctional_edges"] == (
        audit["n_junction_edges"]
    )


def test_young_dupre_local_is_worse_on_a_real_mesh_falsified_result(mesh):
    """FALSIFIED, pinned. Per-edge Young-Dupre loses to the averaged form on a real mesh.

    Recorded as an assertion rather than prose so the negative result cannot quietly rot. The
    mechanism: at equilibrium the true wedge angles are constant
    along a triple line, so all along-line variation is error, averaging the angles filters it
    and taking the cosine first rectifies it into a bias of order `Var(angle) / 2`.
    """
    truth_path = require_substitute_mesh().parent / "000_dict_tensions.npy"
    if not truth_path.exists():
        pytest.skip("ground-truth tensions for the substitute mesh are not available locally")
    truth = np.load(truth_path, allow_pickle=True).item()

    def matched_median(dict_tensions):
        scale = m.gauge_matching_scale(dict_tensions, truth)
        return m.tension_relative_errors({k: scale * v for k, v in dict_tensions.items()}, truth)["median"]

    averaged = matched_median(infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre))
    local = matched_median(infer_tensions(mesh, 1, TensionComputationMethod.YoungDupreLocal))
    assert local > averaged
    # And the naive `l_e / L` weighting is worse still than the line-preserving default, which is
    # why the default is not the specification's literal reading.
    naive = matched_median(
        infer_tensions_young_dupre_local(mesh, 1.0, weighting=PerEdgeWeighting.Edge).dict_tensions,
    )
    assert naive > local


def test_equilibrium_defaults_are_unchanged_and_the_alternatives_are_measurable(mesh):
    """`Equilibrium`'s two defect-cleanup non-changes stay non-changes, but are now switchable and measured."""
    _, shipped, _ = _infer_tension_equilibrium(mesh, 1)
    _, default_flags, _ = _infer_tension_equilibrium(mesh, 1, deduplicate=False, length_weighted=False)
    for key in shipped:
        assert shipped[key] == pytest.approx(default_flags[key], rel=1e-14)

    _, deduplicated, _ = _infer_tension_equilibrium(mesh, 1, deduplicate=True)
    _, weighted, _ = _infer_tension_equilibrium(mesh, 1, length_weighted=True)
    scale = max(abs(value) for value in shipped.values())
    # Since 0.7.0 the gauge is exact, and the uniform triplication of homogeneous rows is a true
    # no-op (to rounding). Until then it was a real change, only because the soft gauge row of ones
    # was not triplicated with them.
    assert max(abs(deduplicated[key] - shipped[key]) for key in shipped) <= 1e-12 * scale
    # The length weighting reweights the blocks against each other: a real change.
    assert max(abs(weighted[key] - shipped[key]) for key in shipped) > 1e-6
