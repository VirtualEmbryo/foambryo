"""Structural properties of the tension matrices, and a smoke test over every method.

The defect-cleanup work, the `foambryo` half: three defects found by a code review — the dead
**`faces_material` table**, the **cotangent operator-precedence defect**, and the **`Equilibrium`
crash**. Three things are pinned here, in increasing scope:

1. **The cotangent identities** (`test_cotan_row_*`) — pure algebra, no mesh involved:
   the row each cotangent builder emits vanishes identically on a junction that is at
   equilibrium. This is what makes the precedence defect a defect rather than a
   different-but-defensible weighting.
2. **Row linearity in the junction-length weight** (`test_every_row_scales_linearly_in_factor`)
   — the property test the precedence defect calls for, run over *all seven* angle-based builders on the
   golden-master mesh. `test_the_linearity_property_test_discriminates` re-implements the precedence defect's
   old expression inline and shows it violates the property, so the check is not vacuous.
3. **Every `TensionComputationMethod` runs to completion** (`test_every_tension_method_*`).
   The `Equilibrium` crash was exactly this class of defect: `Equilibrium` raised `AttributeError` on every
   input, for an unknown length of time, while eight sibling methods worked. Nothing here
   asserts *accuracy* for all nine — only that each one produces a finite tension for every
   interface. The two accuracy assertions that do appear (`test_cotan_methods_are_accurate_*`)
   are thresholds the code before the precedence fix failed by two orders of magnitude.

The deleted `faces_material` table was dead *and*, when the defect-cleanup work fixed it, was on no default code
path: at that time `DcelData.compute_volume_derivatives` used the autodiff route and the closed
form the table lived in was reached only through `compute_volume_derivatives_slow`, so no golden master
covered it. (The sparse-assembly work later made the closed form the default and the variational solve consume it
sparsely — so it *is* now on the default path — but this invariant test predates that and stands
on its own.) `test_volume_derivatives_satisfy_euler_identity` gives it an invariant of its own.
"""

import numpy as np
import pytest

from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions
from foambryo.geometry import compute_volume_cells, compute_volume_derivative_dict
from foambryo.tension_inference import (
    _build_matrix_tension_cotan,
    _build_matrix_tension_inv_cotan,
    _build_matrix_tension_inv_lami,
    _build_matrix_tension_lami,
    _build_matrix_tension_lami_log,
    _build_matrix_tension_projection_yd,
    _build_matrix_tension_symmetrical_yd,
    _compute_z,
    _factors_z,
)
from tests.conftest import DATASET_DIR, require_substitute_mesh

# Every builder that takes the (dict_rad, dict_areas, dict_length_tri, mean_tension)
# signature, i.e. every builder whose rows carry the junction-length weight `factor`.
# `_build_matrix_equilibrium_tensions` is deliberately absent: it takes the mesh itself and
# carries no length weight at all (see test_equilibrium_matrix_is_unweighted).
ANGLE_BUILDERS = [
    _build_matrix_tension_symmetrical_yd,
    _build_matrix_tension_projection_yd,
    _build_matrix_tension_cotan,
    _build_matrix_tension_inv_cotan,
    _build_matrix_tension_lami,
    _build_matrix_tension_inv_lami,
    _build_matrix_tension_lami_log,
]


@pytest.fixture(scope="module")
def golden_mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


@pytest.fixture(scope="module")
def benchmark_ground_truth():
    """Ground-truth tensions for the golden-master mesh (`000_dict_tensions.npy`)."""
    path = DATASET_DIR / "000_dict_tensions.npy"
    if not path.exists():
        pytest.skip(f"{path} not found (sibling benchmarking-dataset/ missing)")
    return np.load(path, allow_pickle=True).item()


@pytest.fixture(scope="module")
def junction_data(golden_mesh):
    """(dict_rad, dict_areas, dict_length_tri) for the golden-master mesh.

    `unique=False` matches every production call site (`geometry.compute_angles_tri`'s
    `unique` flag does not change the mean angles or the length ratios).
    """
    dict_rad, _, dict_length = golden_mesh.compute_angles_tri(unique=False)
    return dict_rad, golden_mesh.compute_areas_interfaces(), dict_length


def _equilibrium_junction(rng):
    """A triple junction that is exactly at mechanical equilibrium.

    Returns `(phi, gamma)` with the three wedge angles summing to `2 pi` and tensions
    satisfying Lami's relation `gamma_i / sin(phi_i) = const`, `phi_i` being the wedge
    *opposite* `gamma_i`. Angles are kept in `(0.2, pi)` so that a Neumann triangle exists
    and `1 - cos(phi)` stays well away from 0.
    """
    while True:
        phi = rng.uniform(0.6, 2.6, 3)
        phi *= 2 * np.pi / phi.sum()
        if np.all(phi < np.pi) and np.all(phi > 0.2):
            gamma = np.sin(phi)
            return phi, gamma / gamma.mean()


@pytest.mark.parametrize("trial", range(20))
def test_cotan_row_vanishes_at_equilibrium(trial):
    """`-(s_i + f_i) g_i + s_i (g_j + g_k) = 0` on an exactly balanced junction.

    This is the identity `_build_matrix_tension_cotan`'s docstring claims, and the reason its
    row must be weighted *as a whole*: the identity is homogeneous of degree 1 in the
    tensions, so `factor * row` still vanishes, while `row` with one term left unweighted
    does not.
    """
    phi, gamma = _equilibrium_junction(np.random.default_rng(trial))
    z1, z2, z3 = _compute_z(*phi)
    f1, f2, f3, s1, s2, s3 = _factors_z(z1, z2, z3)
    g1, g2, g3 = gamma

    for s_i, f_i, g_i, g_j, g_k in [
        (s1, f1, g1, g2, g3),
        (s2, f2, g2, g1, g3),
        (s3, f3, g3, g1, g2),
    ]:
        residual = -(s_i + f_i) * g_i + s_i * (g_j + g_k)
        scale = max(abs(s_i + f_i) * abs(g_i), abs(s_i) * (abs(g_j) + abs(g_k)))
        assert abs(residual) / scale < 1e-12


@pytest.mark.parametrize("trial", range(20))
def test_inv_cotan_row_vanishes_at_equilibrium(trial):
    """The reciprocal identity `-(1/s_i + 1/f_i) g_i + (1/f_i)(g_j + g_k) = 0`."""
    phi, gamma = _equilibrium_junction(np.random.default_rng(100 + trial))
    z1, z2, z3 = _compute_z(*phi)
    f1, f2, f3, s1, s2, s3 = _factors_z(z1, z2, z3)
    g1, g2, g3 = gamma

    for s_i, f_i, g_i, g_j, g_k in [
        (s1, f1, g1, g2, g3),
        (s2, f2, g2, g1, g3),
        (s3, f3, g3, g1, g2),
    ]:
        residual = -(1 / s_i + 1 / f_i) * g_i + (1 / f_i) * (g_j + g_k)
        scale = max((1 / abs(s_i) + 1 / abs(f_i)) * abs(g_i), (1 / abs(f_i)) * (abs(g_j) + abs(g_k)))
        assert abs(residual) / scale < 1e-12


@pytest.mark.parametrize("trial", range(10))
def test_old_cotan_row_residual_matches_its_closed_form(trial):
    """The old (pre-precedence-fix) row leaves residual `s_i g_i (factor - 1)` on a balanced junction.

    Quantifies what the defect cost rather than only asserting that it was wrong: the old
    row is an equilibrium identity **only** for `factor = 1`, i.e. only for a junction
    carrying the whole mesh's junction length. The relative over-weighting of the diagonal
    goes like `1 / factor`, so short junctions were the worst affected.
    """
    rng = np.random.default_rng(200 + trial)
    phi, gamma = _equilibrium_junction(rng)
    z1, z2, z3 = _compute_z(*phi)
    _f1, _f2, f3, _s1, _s2, s3 = _factors_z(z1, z2, z3)
    g1, g2, g3 = gamma

    for factor in rng.uniform(0.001, 1.0, 3):
        old_row_residual = (-s3 - f3 * factor) * g3 + s3 * factor * g1 + s3 * factor * g2
        assert old_row_residual == pytest.approx(s3 * g3 * (factor - 1), rel=1e-9)


def _row_scaling_ratios(row_a, row_b, tol=1e-12):
    """Ratios `row_b[k] / row_a[k]` over the support of `row_a`, or None if not parallel.

    Two rows are parallel iff every such ratio is the same number — which is exactly the
    statement "the row is `factor` times a vector of angle-only coefficients".
    """
    support = np.abs(row_a) > tol * max(np.abs(row_a).max(), 1.0)
    if not support.any():
        return None
    ratios = row_b[support] / row_a[support]
    return ratios


@pytest.mark.parametrize("builder", ANGLE_BUILDERS, ids=lambda b: b.__name__)
def test_every_row_scales_linearly_in_factor(builder, junction_data):
    """Every physics row of every angle-based builder is proportional to its `factor`.

    The property test for the precedence defect. Built twice from the same angles and areas but different
    junction lengths: since `factor = length[junction] / total_length` is the only thing the
    lengths feed, each row of the second matrix must be the corresponding row of the first
    times one scalar. A row with a term that escapes `factor` is not parallel to itself
    under a change of weight — that is precisely the precedence defect.
    """
    dict_rad, dict_areas, dict_length = junction_data
    rng = np.random.default_rng(0)
    perturbed_length = {key: value * rng.uniform(0.5, 2.0) for key, value in dict_length.items()}

    matrix_a, _ = builder(dict_rad, dict_areas, dict_length)
    matrix_b, _ = builder(dict_rad, dict_areas, perturbed_length)
    assert matrix_a.shape == matrix_b.shape

    # The last row is the soft mean-tension normalisation (all ones, weight-free by design).
    observed_scalars = []
    for r in range(matrix_a.shape[0] - 1):
        ratios = _row_scaling_ratios(matrix_a[r], matrix_b[r])
        if ratios is None:  # an all-zero row: nothing to check
            continue
        np.testing.assert_allclose(
            ratios,
            ratios[0],
            rtol=1e-10,
            err_msg=f"{builder.__name__} row {r} is not proportional to its junction-length weight",
        )
        observed_scalars.append(ratios[0])

    assert observed_scalars, f"{builder.__name__} produced no non-empty physics row"
    # Guard against the property holding vacuously because the builder ignores `factor`
    # altogether: the per-junction weights really do differ from one another.
    assert np.std(observed_scalars) > 1e-6, f"{builder.__name__} rows do not depend on the junction length at all"


def test_the_linearity_property_test_discriminates(junction_data):
    """The old (pre-precedence-fix) cotangent row fails `test_every_row_scales_linearly_in_factor`.

    Re-implements the pre-fix expression here rather than relying on a claim about a commit,
    so this file demonstrates on its own that the property above has teeth.
    """
    dict_rad, dict_areas, dict_length = junction_data
    rng = np.random.default_rng(0)
    perturbed_length = {key: value * rng.uniform(0.5, 2.0) for key, value in dict_length.items()}

    def build_with_old_precedence(length):
        """`_build_matrix_tension_cotan`, but with the old `-s3 - f3 * factor` diagonals."""
        matrix, _vector = _build_matrix_tension_cotan(dict_rad, dict_areas, length)
        # Rebuild only the diagonals, the way the old code did. Recovering `factor` per
        # junction from the matrix is not needed: the off-diagonal entries are `s_i * factor`
        # and are already correct, so `s_i * factor` and `f_i * factor` are read back from
        # the row itself and the diagonal is re-formed as `-s_i - f_i * factor`.
        total = sum(length.values()) / 3
        keys = np.array(list(dict_rad.keys()))
        keys = -np.sort(-keys, axis=1)
        ka = np.amax(keys)
        linear = np.unique(keys[:, 0] * ka**2 + keys[:, 1] * ka + keys[:, 2])
        for i, key in enumerate(linear):
            c = key // ka**2
            b = (key - (c * (ka**2))) // ka
            a = key % ka
            z1, z2, z3 = _compute_z(dict_rad[(b, a, c)], dict_rad[(a, b, c)], dict_rad[(a, c, b)])
            f1, f2, f3, s1, s2, s3 = _factors_z(z1, z2, z3)
            factor = length[(a, b, c)] / total
            keys_t = np.array(sorted(dict_areas.keys()))
            nb_zones = np.amax(keys_t) + 1
            reverse = dict(
                zip(keys_t[:, 0] * nb_zones + keys_t[:, 1], np.arange(len(keys_t)), strict=False),
            )
            idx_tab = reverse[min(a, b) * nb_zones + max(a, b)]
            idx_tbc = reverse[min(b, c) * nb_zones + max(b, c)]
            idx_tac = reverse[min(a, c) * nb_zones + max(a, c)]
            matrix[3 * i, idx_tab] = -s3 - f3 * factor
            matrix[3 * i + 1, idx_tbc] = -s1 - f1 * factor
            matrix[3 * i + 2, idx_tac] = -s2 - f2 * factor
        return matrix

    old_a = build_with_old_precedence(dict_length)
    old_b = build_with_old_precedence(perturbed_length)

    non_parallel_rows = 0
    for r in range(old_a.shape[0] - 1):
        ratios = _row_scaling_ratios(old_a[r], old_b[r])
        if ratios is None:
            continue
        if not np.allclose(ratios, ratios[0], rtol=1e-10):
            non_parallel_rows += 1

    assert non_parallel_rows > 0, "the old precedence should violate row linearity in `factor`"


def test_equilibrium_matrix_is_unweighted(golden_mesh):
    """`Equilibrium` applies no junction-length weight, unlike the seven angle builders.

    Recorded rather than fixed: adding a weight would change a public method's output, which
    is outside the defect-cleanup work's remit (it is defect cleanup, and the `Equilibrium` fix is
    "make it run"). It is why
    `_build_matrix_equilibrium_tensions` is excluded from `ANGLE_BUILDERS`, and it is the
    same per-edge/per-line weighting question the per-edge Young-Dupre work opens.
    """
    from foambryo.tension_inference import _build_matrix_equilibrium_tensions

    matrix, _vector = _build_matrix_equilibrium_tensions(golden_mesh, golden_mesh.compute_areas_interfaces())

    # The rows come in blocks of 3 — one vector equation `sum_faces gamma_face * t_face = 0`
    # per trijunctional edge — so the invariant is per (block, interface): the 3-vector each
    # interface contributes to a block is a *unit* direction. A junction-length weight would
    # scale it; nothing does.
    n_blocks = (matrix.shape[0] - 1) // 3
    checked = 0
    for block in range(n_blocks):
        rows = matrix[3 * block : 3 * block + 3]
        for column in np.flatnonzero(np.abs(rows).sum(axis=0) > 0):
            assert np.linalg.norm(rows[:, column]) == pytest.approx(1.0, abs=1e-9)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("mode", list(TensionComputationMethod), ids=lambda m: m.name)
def test_every_tension_method_runs_to_completion(mode, golden_mesh):
    """No member of `TensionComputationMethod` may raise, or return a non-finite tension.

    The `Equilibrium` crash's regression guard. `Equilibrium` raised `AttributeError: 'HalfEdge' object has no
    attribute 'next'` on every input — including the golden-master mesh — and the failure sat
    unnoticed in a committed benchmark record. This test is the reason that cannot recur
    silently for any of the nine.
    """
    expected_interfaces = set(golden_mesh.compute_areas_interfaces())

    tensions = infer_tensions(golden_mesh, mean_tension=1, mode=mode)

    assert set(tensions) == expected_interfaces
    values = np.array(list(tensions.values()))
    assert np.all(np.isfinite(values)), f"{mode.name} returned non-finite tensions"
    assert np.all(values > 0), f"{mode.name} returned a non-positive tension"


@pytest.mark.parametrize("mode", list(TensionComputationMethod), ids=lambda m: m.name)
def test_every_tension_method_returns_residuals(mode, golden_mesh):
    """The `*_and_residuals` entry point works for every method too."""
    from foambryo.tension_inference import infer_tensions_and_residuals

    tensions_array, dict_tensions, _residuals = infer_tensions_and_residuals(golden_mesh, 1, mode)

    assert len(tensions_array) >= len(dict_tensions)
    assert np.all(np.isfinite(np.asarray(tensions_array, dtype=float)[: len(dict_tensions)]))


@pytest.mark.parametrize("mode", [TensionComputationMethod.Cotan, TensionComputationMethod.InvCotan])
def test_cotan_methods_are_accurate_after_the_precedence_fix(mode, golden_mesh, benchmark_ground_truth):
    """The precedence fix's acceptance criterion, as a number rather than a claim.

    Before the fix, median relative tension error on this mesh was 3.7e-01 (`Cotan`) and
    6.1e-01 (`InvCotan`), and `InvCotan` returned `mean(gamma) = 0.42` — the badly scaled
    diagonal was overpowering the soft normalisation row. Both now sit at ~2e-03 with
    `mean(gamma) = 1`. The thresholds below are two orders of magnitude tighter than the
    old behaviour and two orders looser than the new, so they pin the fix without pinning
    library-version floating-point noise.
    """
    truth = benchmark_ground_truth
    tensions = infer_tensions(golden_mesh, mean_tension=1, mode=mode)

    keys = sorted(truth)
    inferred = np.array([tensions[key] for key in keys])
    expected = np.array([truth[key] for key in keys])

    assert np.mean(inferred) == pytest.approx(1.0, abs=1e-3)
    assert np.median(np.abs(inferred - expected) / np.abs(expected)) < 1e-2


def test_volume_derivatives_satisfy_euler_identity(golden_mesh):
    """`sum_i x_i . dV/dx_i == 3 V` for every cell (the `faces_material` fix's invariant).

    A cell volume is a homogeneous degree-3 function of the vertex coordinates, so Euler's
    homogeneous-function theorem pins the closed-form derivative dictionary against the
    independently computed volumes. This covers the function the dead `faces_material` table
    lived in, which no golden master reaches (`compute_volume_derivatives` uses autodiff).
    """
    derivatives = compute_volume_derivative_dict(golden_mesh)
    volumes = compute_volume_cells(golden_mesh)

    assert set(derivatives) == set(volumes)
    for cell, derivative in derivatives.items():
        assert float(np.sum(golden_mesh.v * derivative)) == pytest.approx(3 * volumes[cell], rel=1e-10)
