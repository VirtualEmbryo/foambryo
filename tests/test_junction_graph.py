"""Tests for `foambryo.junction_graph` (the junction-graph work).

Per the project's testing discipline, the **positive controls come first**: a mesh
that provably fragments (two spatially-separated doublets) and a mesh with a provably closed
interface, both checked against a hand-derived expected partition, before any negative result
elsewhere in this project's test suite is trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import linalg

from foambryo_benchmarks import phantoms as ph
from foambryo.dcel import DcelData
from foambryo.junction_graph import (
    partition_by_junction_graph,
    partition_from_mesh,
    require_single_component_solution,
    solve_gauged_least_squares,
)
from foambryo.per_edge_tension_inference import infer_tensions_young_dupre_local
from foambryo.tension_inference import (
    TensionComputationMethod,
    infer_tensions,
    infer_tensions_and_residuals,
    infer_tensions_by_junction_component,
)


# =====================================================================
# Mesh builders: two separated doublets, and an isolated closed sphere
# =====================================================================
def _relabel(regions: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = regions.copy()
    for old, new in mapping.items():
        out[regions == old] = new
    return out


def _combine(*parts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate several `(V, F, regions)` meshes into one, offsetting vertex indices."""
    vs, fs, rs = [], [], []
    offset = 0
    for v, f, r in parts:
        vs.append(v)
        fs.append(f + offset)
        rs.append(r)
        offset += len(v)
    return np.vstack(vs), np.vstack(fs), np.vstack(rs)


def _two_separated_doublets(ratio_a: float = 1.0, ratio_b: float = 0.6) -> tuple[DcelData, dict, dict]:
    """Two analytic doublets, translated far apart, cell labels `{1,2}` and `{3,4}`.

    They share only the exterior label `0`; since they never touch, `0`'s two boundary
    components never share a triple line, so the combined mesh's junction graph must have
    exactly two components: `{(0,1),(0,2),(1,2)}` and `{(0,3),(0,4),(3,4)}`.

    Returns:
        tuple[DcelData, dict, dict]: the mesh, and each doublet's `mean_tension=1`
            ground-truth tensions (relabelled to match the combined mesh).
    """
    va, fa, ra = ph.doublet_mesh(ratio_a, r0=1.0, n_rings=24, n_phi=48)
    vb, fb, rb = ph.doublet_mesh(ratio_b, r0=1.0, n_rings=24, n_phi=48)
    vb = vb + np.array([0.0, 0.0, 50.0])  # far beyond either doublet's own radius
    rb = _relabel(rb, {1: 3, 2: 4})

    v, f, r = _combine((va, fa, ra), (vb, fb, rb))
    mesh = DcelData(v, f, r)

    truth_a = ph.doublet_ground_truth_tensions(ratio_a, mean_tension=1.0)
    truth_b_raw = ph.doublet_ground_truth_tensions(ratio_b, mean_tension=1.0)
    truth_b = {
        (0, 3): truth_b_raw[(0, 1)],
        (0, 4): truth_b_raw[(0, 2)],
        (3, 4): truth_b_raw[(1, 2)],
    }
    return mesh, truth_a, truth_b


def _uv_sphere(
    label_outside: int,
    label_inside: int,
    center: np.ndarray,
    radius: float,
    n_theta: int = 10,
    n_phi: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed UV-sphere, `(V, F, regions)`, oriented outward from `label_outside`.

    Standalone (touches nothing else), so its one interface `(label_outside,
    label_inside)` (after sorting) never appears in any triple line -- the "closed
    interface" case `foambryo.junction_graph`'s module docstring describes.
    """
    thetas = np.linspace(0, np.pi, n_theta + 1)
    phis = 2 * np.pi * np.arange(n_phi) / n_phi

    verts = [center + radius * np.array([0.0, 0.0, 1.0])]
    verts.extend(
        center + radius * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        for theta in thetas[1:-1]
        for phi in phis
    )
    verts.append(center + radius * np.array([0.0, 0.0, -1.0]))
    verts = np.array(verts)

    n_rings = n_theta - 1

    def ring_start(i: int) -> int:
        return 1 + i * n_phi

    top, bottom = 0, len(verts) - 1

    faces = [[top, ring_start(0) + j, ring_start(0) + (j + 1) % n_phi] for j in range(n_phi)]
    for i in range(n_rings - 1):
        for j in range(n_phi):
            a = ring_start(i) + j
            b = ring_start(i) + (j + 1) % n_phi
            c = ring_start(i + 1) + j
            d = ring_start(i + 1) + (j + 1) % n_phi
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces.extend(
        [bottom, ring_start(n_rings - 1) + (j + 1) % n_phi, ring_start(n_rings - 1) + j] for j in range(n_phi)
    )

    faces = np.array(faces)
    # Outward normal check on the first face; flip every face if it points inward, since
    # this construction is either globally outward or globally inward.
    p0, p1, p2 = verts[faces[0]]
    normal = np.cross(p1 - p0, p2 - p0)
    if np.dot(normal, p0 - center) < 0:
        faces = faces[:, [0, 2, 1]]

    lo, hi = min(label_outside, label_inside), max(label_outside, label_inside)
    regions = np.tile(np.array([lo, hi]), (len(faces), 1))
    return verts, faces, regions


# =====================================================================
# Positive control 1: provable fragmentation
# =====================================================================
def test_two_separated_doublets_partition_has_exactly_two_components():
    mesh, _truth_a, _truth_b = _two_separated_doublets()
    partition = partition_from_mesh(mesh)

    assert partition.fragmented
    assert partition.n_components == 2
    assert partition.isolated == ()
    sizes = sorted(len(component) for component in partition.components)
    assert sizes == [3, 3]

    interfaces_a = {(0, 1), (0, 2), (1, 2)}
    interfaces_b = {(0, 3), (0, 4), (3, 4)}
    found = {frozenset(component) for component in partition.components}
    assert found == {frozenset(interfaces_a), frozenset(interfaces_b)}


@pytest.mark.parametrize(
    "mode",
    [
        TensionComputationMethod.YoungDupre,
        TensionComputationMethod.ProjectionYoungDupre,
        TensionComputationMethod.Cotan,
        TensionComputationMethod.InvCotan,
        TensionComputationMethod.Lami,
        TensionComputationMethod.InvLami,
        TensionComputationMethod.LogLami,
        TensionComputationMethod.YoungDupreLocal,
    ],
)
def test_two_separated_doublets_solved_per_component_matches_ground_truth(mode):
    """The core positive control: each component recovers ITS OWN doublet's tensions.

    Before the junction-graph work, `scipy.linalg.lstsq`'s minimum-norm tie-break would drive one
    component's tensions toward zero (a gauge-matched relative error of exactly 1.0);
    with the fix each component is solved with its own `mean(gamma) = 1` gauge and should
    recover its own analytic doublet answer to near machine precision (these are exact
    analytic meshes, not reconstructions).
    """
    mesh, truth_a, truth_b = _two_separated_doublets()
    solution = infer_tensions_by_junction_component(mesh, mean_tension=1.0, mode=mode)

    assert solution.fragmented
    assert solution.n_components == 2
    assert solution.undetermined == ()
    assert set(solution.component_of) == {(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)}
    # The two components must map to different component ids.
    assert solution.component_of[(0, 1)] == solution.component_of[(1, 2)]
    assert solution.component_of[(0, 3)] == solution.component_of[(3, 4)]
    assert solution.component_of[(0, 1)] != solution.component_of[(0, 3)]
    assert len(solution.gauge_notes) == 2

    for value in solution.dict_tensions.values():
        assert not math.isnan(value)

    # LogLami's log-domain regression carries more discretisation error than the other
    # methods at this mesh resolution -- a pre-existing property of the method (not
    # something the junction-graph work touches), so it gets a looser tolerance rather than a finer mesh
    # slowing every other case down.
    tolerance = 0.15 if mode is TensionComputationMethod.LogLami else 0.08
    for truth in (truth_a, truth_b):
        for key, expected in truth.items():
            assert solution.dict_tensions[key] == pytest.approx(expected, abs=tolerance)


def test_two_separated_doublets_default_infer_tensions_is_also_fixed():
    """The pre-existing public entry point must be fixed too, not only the new API.

    `infer_tensions`/`infer_tensions_and_residuals` route through the same
    `solve_gauged_least_squares` call as `infer_tensions_by_junction_component` (the junction-graph work),
    so a fragmented mesh must no longer come back pinned at a gauge-matched error of 1.0 on
    either component.
    """
    mesh, truth_a, truth_b = _two_separated_doublets()
    dict_tensions = infer_tensions(mesh, mean_tension=1.0, mode=TensionComputationMethod.YoungDupre)

    for truth in (truth_a, truth_b):
        for key, expected in truth.items():
            assert dict_tensions[key] == pytest.approx(expected, abs=0.08)

    tensions, dict_tensions_2, residuals = infer_tensions_and_residuals(
        mesh,
        mean_tension=1.0,
        mode=TensionComputationMethod.YoungDupre,
    )
    assert dict_tensions_2 == dict_tensions
    assert not np.any(np.isnan(tensions))
    assert residuals.size >= 0  # concatenated per-component residuals, never crashes


# =====================================================================
# Positive control 2: a provably closed interface (undetermined, never a number)
# =====================================================================
def test_isolated_sphere_is_reported_as_undetermined_not_a_component():
    v_doublet, f_doublet, r_doublet = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    v_sphere, f_sphere, r_sphere = _uv_sphere(
        label_outside=0,
        label_inside=5,
        center=np.array([0.0, 0.0, 50.0]),
        radius=1.0,
    )
    v, f, r = _combine((v_doublet, f_doublet, r_doublet), (v_sphere, f_sphere, r_sphere))

    partition = partition_by_junction_graph(f, r)
    assert partition.fragmented
    assert partition.isolated == ((0, 5),)
    assert partition.n_components == 1
    assert partition.components == (((0, 1), (0, 2), (1, 2)),)
    assert (0, 5) not in partition.component_of

    mesh = DcelData(v, f, r)
    solution = infer_tensions_by_junction_component(mesh, mean_tension=1.0, mode=TensionComputationMethod.YoungDupre)
    assert solution.undetermined == ((0, 5),)
    assert math.isnan(solution.dict_tensions[(0, 5)])
    assert solution.component_of[(0, 5)] is None
    for key in ((0, 1), (0, 2), (1, 2)):
        assert not math.isnan(solution.dict_tensions[key])

    with pytest.raises(ValueError, match="refusing to aggregate"):
        require_single_component_solution(solution)


def test_isolated_sphere_never_reported_as_a_number_by_the_default_api():
    """`infer_tensions` must never silently turn an undetermined interface into `0.0`."""
    v_doublet, f_doublet, r_doublet = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    v_sphere, f_sphere, r_sphere = _uv_sphere(0, 5, np.array([0.0, 0.0, 50.0]), 1.0)
    v, f, r = _combine((v_doublet, f_doublet, r_doublet), (v_sphere, f_sphere, r_sphere))
    mesh = DcelData(v, f, r)

    dict_tensions = infer_tensions(mesh, mean_tension=1.0, mode=TensionComputationMethod.YoungDupre)
    assert math.isnan(dict_tensions[(0, 5)])


# =====================================================================
# Non-fragmented meshes: backward compatibility (bit-identical), the doublet itself
# =====================================================================
def test_single_doublet_is_not_fragmented_and_has_one_gauge_note():
    v, f, r = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    partition = partition_by_junction_graph(f, r)
    assert not partition.fragmented
    assert partition.n_components == 1
    assert partition.isolated == ()

    mesh = DcelData(v, f, r)
    solution = infer_tensions_by_junction_component(mesh, mean_tension=1.0, mode=TensionComputationMethod.YoungDupre)
    assert not solution.fragmented
    assert solution.n_components == 1
    assert len(solution.gauge_notes) == 1
    assert require_single_component_solution(solution) == 0


# =====================================================================
# Unit tests on solve_gauged_least_squares directly, against hand-built systems
# =====================================================================
def _partition_two_pairs():
    from foambryo.junction_graph import JunctionGraphPartition

    return JunctionGraphPartition(
        interfaces=((0, 1), (2, 3), (4, 5), (6, 7)),
        components=(((0, 1), (2, 3)), ((4, 5), (6, 7))),
        component_of={(0, 1): 0, (2, 3): 0, (4, 5): 1, (6, 7): 1},
        isolated=(),
    )


def test_solve_gauged_least_squares_not_fragmented_is_bit_identical_to_lstsq():
    from foambryo.junction_graph import JunctionGraphPartition

    rng = np.random.default_rng(0)
    keys = [(0, 1), (0, 2), (1, 2)]
    matrix_m = rng.normal(size=(7, 3))
    matrix_m[-1] = 1.0
    vector_b = rng.normal(size=7)
    vector_b[-1] = 3.0

    partition = JunctionGraphPartition(
        interfaces=tuple(keys),
        components=(tuple(keys),),
        component_of=dict.fromkeys(keys, 0),
        isolated=(),
    )
    solution = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    expected, expected_resid, _rank, _sv = linalg.lstsq(matrix_m, vector_b)
    assert np.array_equal(solution.tensions, expected)
    assert np.array_equal(solution.lstsq_residuals, expected_resid)
    assert not solution.fragmented


def test_solve_gauged_least_squares_fragmented_solves_each_block_with_its_own_gauge():
    keys = [(0, 1), (2, 3), (4, 5), (6, 7)]
    # Two independent 2x2 identity blocks, each pinned by mean(gamma) = 1 -> both columns
    # of a block must equal 1 exactly once decomposed (an exactly-determined 2x2 system).
    matrix_m = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],  # the global (wrong) gauge row this decomposition ignores
        ],
    )
    vector_b = np.array([0.0, 0.0, 4.0])  # nm * mean_tension = 4 * 1.0

    partition = _partition_two_pairs()
    solution = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)

    assert solution.fragmented
    assert solution.n_components == 2
    for key in keys:
        assert solution.dict_tensions[key] == pytest.approx(1.0)
    assert solution.component_of[(0, 1)] == solution.component_of[(2, 3)]
    assert solution.component_of[(4, 5)] == solution.component_of[(6, 7)]
    assert solution.component_of[(0, 1)] != solution.component_of[(4, 5)]


def test_solve_gauged_least_squares_rejects_a_row_spanning_two_components():
    keys = [(0, 1), (2, 3), (4, 5), (6, 7)]
    matrix_m = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],  # spans component 0's (2,3) and component 1's (6,7)
            [1.0, 1.0, 1.0, 1.0],
        ],
    )
    vector_b = np.array([0.0, 0.0, 4.0])
    partition = _partition_two_pairs()
    with pytest.raises(ValueError, match="more than one triple-junction-graph"):
        solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)


def test_solve_gauged_least_squares_requires_matching_interface_sets():
    from foambryo.junction_graph import JunctionGraphPartition

    partition = JunctionGraphPartition(
        interfaces=((0, 1), (0, 2)),
        components=(((0, 1), (0, 2)),),
        component_of={(0, 1): 0, (0, 2): 0},
        isolated=(),
    )
    matrix_m = np.ones((2, 3))
    vector_b = np.ones(2)
    with pytest.raises(ValueError, match="do not match"):
        solve_gauged_least_squares(matrix_m, vector_b, [(0, 1), (0, 2), (0, 3)], 1.0, partition)


# =====================================================================
# Scope: excluded methods must raise, not silently misapply the decomposition
# =====================================================================
@pytest.mark.parametrize(
    "mode",
    [
        TensionComputationMethod.Equilibrium,
        TensionComputationMethod.ForceBalance,
        TensionComputationMethod.ForceBalanceSoftRow,
        TensionComputationMethod.ForceBalanceMAP,
        TensionComputationMethod.Variational,
    ],
)
def test_infer_tensions_by_junction_component_refuses_unsupported_methods(mode):
    v, f, r = ph.doublet_mesh(1.0, r0=1.0, n_rings=6, n_phi=10)
    mesh = DcelData(v, f, r)
    with pytest.raises(ValueError, match="not decomposed"):
        infer_tensions_by_junction_component(mesh, mean_tension=1.0, mode=mode)


def test_young_dupre_local_result_carries_component_fields_on_a_healthy_mesh():
    v, f, r = ph.doublet_mesh(1.0, r0=1.0, n_rings=6, n_phi=10)
    mesh = DcelData(v, f, r)
    result = infer_tensions_young_dupre_local(mesh, mean_tension=1.0)
    assert not result.fragmented
    assert result.n_components == 1
    assert result.undetermined == ()
    assert set(result.component_of) == {(0, 1), (0, 2), (1, 2)}
