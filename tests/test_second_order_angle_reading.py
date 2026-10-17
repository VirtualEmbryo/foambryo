"""Tests for the second-order (quadric) trijunction angle reading — the second-order-angle-reading work.

What these tests pin, and why each one exists
----------------------------------------------
1. **The bit-identity gate.** The second-order-angle-reading work added a `reading=` selector to the
one function that defines
   `foambryo`'s trijunction angle convention. The default must produce, bit for bit, what
   commit `55bac17` produced — not "the same to a tolerance". Every published tension in
   this repository, every benchmark row and every figure depends on that. The baseline in
   `tests/data/angle_reading_bit_identity_baseline.json` is a SHA-256 digest of the raw IEEE-754
   bits of every angle, length and tension the source produced before this work, on all 47
   `benchmarking-dataset` reference meshes, generated from a `git archive` export of the branch
   point. A one-ulp drift anywhere fails this test. Bit identity holds only on the numerical stack
   it was verified on (numpy and scipy, `BITWISE_NUMPY`, `BITWISE_SCIPY`); elsewhere that test skips
   with the reason, and a second gate compares the same quantities with the stored numbers of
   `tests/data/angle_reading_reference_values.json` to a relative 1e-12, on every stack.

2. **Safeguards 1, 2 and 4** of the inherited five (see
   `foambryo.second_order_angle_reading`'s module docstring, and dw3d's chord-versus-tangent
   and curvature-from-data work):
   the estimator sees nothing but the mesh; the fit is a generic quadric; and a zero curvature
   returns the chord angle **bitwise**.

3. **The fallback is visible.** A mesh the quadric cannot fully cover must say so, in the
   returned report and — on the report-less convenience paths — as a warning.

4. **The hyperparameters are not reachable from a reading.** `window_factor` and `axial_factor`
   are module constants at the curvature-from-data work's a-priori values, selected by a later
   stage of the identifiability-floor work under a pre-registered coverage floor. Nothing a
   `reading=` selector reaches, and no inference entry point, lets a caller tune them against the
   contact-angle flattening slope, the tension contrast or the per-line angle error, which is
   safeguard 5. `quadric_angle_rotations` itself accepts them keyword-only so the cost of the window
   can be measured without a second implementation of the fit; at the shipped values that call is
   bitwise the default one, executed here rather than asserted.
"""

import functools
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import (
    ANGLE_READING_METHODS,
    CHORD_READING,
    QUADRIC_READING,
    AngleReadingCoverageWarning,
    TensionComputationMethod,
    dcel_mesh_from_file,
    infer_tensions,
    infer_tensions_and_residuals_with_report,
)
from foambryo.dcel import DcelData
from foambryo.geometry import (
    compute_angles_tri,
    compute_angles_tri_with_report,
    compute_trijunction_edges,
    compute_trijunction_edges_with_report,
)
from foambryo.per_edge_tension_inference import infer_tensions_young_dupre_local
from foambryo.second_order_angle_reading import (
    FIT_CONSTANT_TERM,
    JUNCTION_ENDPOINT_POLICIES,
    JUNCTION_ENDPOINTS,
    QUADRIC_AXIAL_FACTOR,
    QUADRIC_WINDOW_FACTOR,
    AngleReadingReport,
    junction_frames,
    quadric_angle_rotations,
    rotations_for_reading,
)
from tests.conftest import DATASET_DIR, require_substitute_mesh

BASELINE = Path(__file__).resolve().parent / "data" / "angle_reading_bit_identity_baseline.json"
#: The same quantities as plain numbers (every method's tensions per interface; count, sum and sum
#: of squares of the angles and lengths), written by `tests/data/make_angle_reading_reference_values.py`,
#: which refuses to write unless every digest of its run equals `BASELINE`'s. The tolerance gate
#: reads it.
REFERENCE_VALUES = Path(__file__).resolve().parent / "data" / "angle_reading_reference_values.json"
#: The numerical stack the bit-identity digests are verified on. Another numpy or scipy can change
#: the last bits of a solve without any change in foambryo (numpy 2.4.6 / scipy 1.17.1, the newest
#: that install on Python 3.11, move `Variational`'s tensions by up to 1.8e-14 relative), so the
#: bitwise gate runs only on this stack; the tolerance gate runs on every stack.
BITWISE_NUMPY, BITWISE_SCIPY = "2.5.1", "1.18.0"
#: Relative tolerance of the gate that runs on every numerical stack.
REFERENCE_RTOL = 1e-12
#: The digest helper is duplicated from
#: the project's research history rather than
#: imported, deliberately: a permanent test must not depend on a research script that may be
#: pruned before release, and a gate that shares its implementation with the thing it checks is
#: not much of a gate.
ANGLE_METHOD_NAMES = (
    "YoungDupre",
    "ProjectionYoungDupre",
    "Lami",
    "InvLami",
    "LogLami",
    "YoungDupreLocal",
    "Cotan",
    "InvCotan",
)
NON_ANGLE_METHOD_NAMES = ("Variational", "ForceBalance")
#: The methods whose mean-tension gauge became exact in 0.7.0 (`foambryo.junction_graph.solve_with_exact_gauge`):
#: their tensions are the baseline's times one factor per case, by design, so they are compared
#: gauge-matched (the baseline rescaled to mean 1) and left out of the bitwise digests. Their angles
#: are unchanged, and are what these gates are about.
EXACT_GAUGE_SINCE_0_7 = (
    "YoungDupre", "ProjectionYoungDupre", "Lami", "InvLami", "YoungDupreLocal", "Cotan", "InvCotan",
)
#: Methods whose tensions are solved through `solve_with_exact_gauge` since 0.7.0: their last bits
#: moved (a differently weighted gauge row), so they leave the bitwise digests. `LogLami`'s gauge was
#: already exact, so its values are compared to the baseline as they are, to 1e-12.
SOLVED_THROUGH_THE_EXACT_GAUGE = (*EXACT_GAUGE_SINCE_0_7, "LogLami")


def _digest(items):
    import hashlib
    import struct

    hasher = hashlib.sha256()
    for key, value in sorted(items, key=lambda pair: pair[0]):
        hasher.update(repr(key).encode())
        hasher.update(struct.pack("<d", float(value)))
    return hasher.hexdigest()


def _sums(values):
    """Count, sum and sum of squares of a collection of floats: a small, tolerance-checkable record."""
    array = np.asarray(list(values), dtype=np.float64)
    return {"n": int(array.size), "sum": float(array.sum()), "sum_sq": float((array**2).sum())}


@functools.cache
def _measure_one_case(mesh_path):
    """`(digests, values)` of everything the default reading produces on one mesh, computed once.

    `digests` is the bit-identity record (`_digest` of each quantity); `values` holds the same
    quantities as numbers: every method's tensions per interface, and the count, sum and sum of
    squares of the angles, lengths and per-edge angles.
    """
    mesh = dcel_mesh_from_file(str(mesh_path))
    record, values = {}, {}
    for unique in (True, False):
        radians, degrees, lengths = compute_angles_tri(mesh, unique=unique)
        suffix = "unique" if unique else "repeated"
        record[f"angles_rad_{suffix}"] = _digest(list(radians.items()))
        record[f"angles_deg_{suffix}"] = _digest(list(degrees.items()))
        record[f"lengths_{suffix}"] = _digest(list(lengths.items()))
        record[f"n_wedge_keys_{suffix}"] = len(radians)
        values[f"angles_rad_{suffix}"] = _sums(radians.values())
        values[f"lengths_{suffix}"] = _sums(lengths.values())
    edges = compute_trijunction_edges(mesh)
    record["trijunction_edge_angles"] = _digest(
        [((edge.vertices, key), angle) for edge in edges for key, angle in zip(edge.keys, edge.angles, strict=True)],
    )
    record["n_trijunction_edges"] = len(edges)
    values["trijunction_edge_angles"] = _sums(angle for edge in edges for angle in edge.angles)
    for name in (*ANGLE_METHOD_NAMES, *NON_ANGLE_METHOD_NAMES):
        tensions = infer_tensions(mesh, 1, getattr(TensionComputationMethod, name))
        record[f"tensions_{name}"] = _digest(list(tensions.items()))
        values[f"tensions_{name}"] = [[[int(k) for k in key], float(value)] for key, value in sorted(tensions.items())]
    return record, values


def _digest_one_case(mesh_path):
    return _measure_one_case(str(mesh_path))[0]


def _benchmark_meshes():
    if not DATASET_DIR.exists():
        pytest.skip(f"{DATASET_DIR} not found (sibling benchmarking-dataset/ missing)")
    meshes = sorted(DATASET_DIR.glob("*_mesh.rec"))
    assert len(meshes) == 47, f"expected 47 reference meshes, found {len(meshes)}"
    return meshes


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip(f"{BASELINE} not found")
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def reference_mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


# ---------------------------------------------------------------------------------------
# 1 -- the bit-identity gate
# ---------------------------------------------------------------------------------------
def test_default_reading_is_bitwise_unchanged_on_every_benchmark_case(baseline):
    """The default path must reproduce commit `55bac17` bit for bit, on all 47 cases.

    Angles (radians and degrees), accumulated lengths, per-edge trijunction angles, and the
    tensions from every one of the ten inference methods, digested over their raw IEEE-754
    bits. This is the gate the second-order-angle-reading work is accepted or rejected on; if it fails, the fix is
    in the source, never in `tests/data/angle_reading_bit_identity_baseline.json`.

    Bit identity is a property of the whole numerical stack, so this runs only on the numpy and
    scipy it is verified on (`BITWISE_NUMPY`, `BITWISE_SCIPY`) and skips elsewhere, saying why;
    `test_default_reading_matches_the_reference_values_to_1e12` is the gate on every other stack.
    """
    import scipy

    if (np.__version__, scipy.__version__) != (BITWISE_NUMPY, BITWISE_SCIPY):
        pytest.skip(
            f"bit identity is verified on numpy {BITWISE_NUMPY} / scipy {BITWISE_SCIPY}, and this is numpy "
            f"{np.__version__} / scipy {scipy.__version__}: the last bits of a solve may differ; the 1e-12 "
            "tolerance gate (test_default_reading_matches_the_reference_values_to_1e12) still runs",
        )
    meshes = _benchmark_meshes()
    assert set(baseline["cases"]) == {p.stem.removesuffix("_mesh") for p in meshes}

    differences = []
    for path in meshes:
        case = path.stem.removesuffix("_mesh")
        recorded = baseline["cases"][case]
        for field, value in _digest_one_case(path).items():
            if field.removeprefix("tensions_") in SOLVED_THROUGH_THE_EXACT_GAUGE:
                continue
            if recorded[field] != value:
                differences.append(f"{case}/{field}")
    assert not differences, (
        f"the default angle reading is no longer bit-identical to {baseline['source']} on "
        f"{len(differences)} digest(s): {differences[:12]}"
    )


def test_default_reading_matches_the_reference_values_to_1e12():
    """The default path reproduces commit `55bac17`'s numbers to 1e-12, on every numerical stack.

    The companion of the bit-identity gate, for the stacks that gate skips: every method's tension on
    every interface of the 47 cases (relative to the case's largest tension), and the count, sum and
    sum of squares of the angles, lengths and per-edge angles, against
    `tests/data/angle_reading_reference_values.json`. Those values were written in a run whose
    digests all equal the bit-identity baseline, so they are the baseline's own numbers. 1e-12 is far
    above the 1.8e-14 measured between numpy 2.5.1 and 2.4.6, and far below any change of method.
    """
    reference = json.loads(REFERENCE_VALUES.read_text())["cases"]
    meshes = _benchmark_meshes()
    assert set(reference) == {p.stem.removesuffix("_mesh") for p in meshes}
    failures = []
    for path in meshes:
        case = path.stem.removesuffix("_mesh")
        values = _measure_one_case(str(path))[1]
        for field, expected in reference[case].items():
            actual = values[field]
            if field.startswith("tensions_"):
                assert [k for k, _ in actual] == [k for k, _ in expected], f"{case}/{field}: interfaces differ"
                a = np.array([v for _, v in actual], dtype=np.float64)
                e = np.array([v for _, v in expected], dtype=np.float64)
                if field.removeprefix("tensions_") in EXACT_GAUGE_SINCE_0_7:
                    e = e / np.nanmean(e)  # the baseline's soft-row scale, gauge-matched to mean 1
                scale = float(np.nanmax(np.abs(e))) if np.isfinite(e).any() else 1.0
                same_nan = np.array_equal(np.isnan(a), np.isnan(e))
                close = np.allclose(a, e, rtol=REFERENCE_RTOL, atol=REFERENCE_RTOL * scale, equal_nan=True)
                if not (same_nan and close):
                    failures.append(f"{case}/{field}")
            elif actual["n"] != expected["n"] or not all(
                np.isclose(actual[k], expected[k], rtol=REFERENCE_RTOL, atol=0.0) for k in ("sum", "sum_sq")
            ):
                failures.append(f"{case}/{field}")
    assert not failures, f"{len(failures)} quantity(ies) moved by more than {REFERENCE_RTOL:g}: {failures[:12]}"


def test_the_selector_defaults_to_the_chord_reading(reference_mesh):
    """A caller who passes nothing gets the historical convention, not the new one."""
    default = compute_angles_tri(reference_mesh, unique=False)
    explicit = compute_angles_tri(reference_mesh, unique=False, reading=CHORD_READING)
    assert default[0] == explicit[0]
    assert default[2] == explicit[2]


def test_an_unknown_reading_is_refused(reference_mesh):
    with pytest.raises(ValueError, match="unknown angle reading"):
        compute_angles_tri(reference_mesh, reading="second-order")
    with pytest.raises(ValueError, match="unknown angle reading"):
        infer_tensions(reference_mesh, angle_reading="tangent")


def test_a_non_angle_method_refuses_a_non_default_reading(reference_mesh):
    """Silently ignoring `angle_reading` would hand back a result that looks corrected.

    `Variational` and the `ForceBalance*` members assemble an operator rather than reading
    angles, so the selector cannot reach them. Refusing is the honest answer.
    """
    for name in NON_ANGLE_METHOD_NAMES:
        mode = getattr(TensionComputationMethod, name)
        assert mode not in ANGLE_READING_METHODS
        with pytest.raises(ValueError, match="does not build its rows from measured"):
            infer_tensions(reference_mesh, 1, mode, angle_reading=QUADRIC_READING)
        # ... but the default reading is always accepted, for every method.
        infer_tensions(reference_mesh, 1, mode, angle_reading=CHORD_READING)


def test_young_dupre_local_refuses_the_quadric_reading(reference_mesh):
    """The quadric-safeguard work: this combination measured +0.0290 relative RMS tension error at md=3, +0.97 at md=4.

    The per-edge balance never performs the along-line average that cancels the quadric
    correction's per-face scatter, so it eats the scatter directly. Both call paths must refuse
    it: the direct entry point, and `infer_tensions`'s dispatch to it.
    """
    with pytest.raises(ValueError, match="quadric"):
        infer_tensions_young_dupre_local(reference_mesh, angle_reading=QUADRIC_READING)
    with pytest.raises(ValueError, match="quadric"):
        infer_tensions(
            reference_mesh, 1, TensionComputationMethod.YoungDupreLocal, angle_reading=QUADRIC_READING,
        )
    # ... but the default reading is always accepted.
    infer_tensions_young_dupre_local(reference_mesh, angle_reading=CHORD_READING)
    infer_tensions(reference_mesh, 1, TensionComputationMethod.YoungDupreLocal, angle_reading=CHORD_READING)


# ---------------------------------------------------------------------------------------
# 2 -- the inherited safeguards
# ---------------------------------------------------------------------------------------
def test_safeguard_1_the_estimator_takes_a_mesh_and_four_switches_and_nothing_else():
    """No parameter through which a reference mesh, a truth angle or a tension could reach it.

    The estimator takes exactly one mesh-derived argument. The other four are keyword-only
    switches over how the fit reads *this* mesh -- two window multipliers, the membership of the
    frame's own two junction-edge endpoints, and whether the height field carries a constant term.
    None of them can carry information from outside the mesh, which is what safeguard 1 is about.
    The statement that no production path can set them is safeguard 5's, below.

    **The parameter list is asserted exactly, and adding to it is meant to fail here.** A new
    argument on this function is the one place a reference mesh, a target angle or a tension could
    enter the estimator, so it has to be looked at by a person rather than absorbed.
    """
    import inspect

    signature = inspect.signature(quadric_angle_rotations)
    assert list(signature.parameters) == [
        "mesh", "window_factor", "axial_factor", "junction_endpoints", "fit_constant_term",
    ]
    for name, default in (
        ("window_factor", QUADRIC_WINDOW_FACTOR),
        ("axial_factor", QUADRIC_AXIAL_FACTOR),
        ("junction_endpoints", JUNCTION_ENDPOINTS),
        ("fit_constant_term", FIT_CONSTANT_TERM),
    ):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == default
    assert isinstance(signature.parameters["window_factor"].default, float)
    assert isinstance(signature.parameters["axial_factor"].default, float)
    assert signature.parameters["junction_endpoints"].default in JUNCTION_ENDPOINT_POLICIES
    assert isinstance(signature.parameters["fit_constant_term"].default, bool)


def test_safeguard_4_a_zero_rotation_is_the_identity_bitwise(reference_mesh):
    """A fitted curvature of zero must return the chord angle exactly — `==`, not `approx`.

    dw3d's chord-versus-tangent work's own formulation of safeguard 4, and the exact statement the implementation
    makes: `_wedges_at_half_edge` skips the Rodrigues call on a falsy angle, so a rotation map
    that is empty, or explicitly `0.0` (or `-0.0`) everywhere, is the identity to the bit.
    Checked on a real benchmark mesh, not a phantom, so it also covers the key plumbing.
    """
    from foambryo.geometry import _wedges_at_half_edge

    reference, explicit_zero, explicit_negative_zero, empty = {}, {}, {}, {}
    for edge in reference_mesh.half_edges:
        if len(edge.twin) <= 1:
            continue
        geometric = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
        wedges = _wedges_at_half_edge(mesh=reference_mesh, edge=edge)
        if wedges is None:
            continue
        zeros = dict.fromkeys(((geometric, int(face.key)) for face in _incident_faces(reference_mesh, edge)), 0.0)
        minus = dict.fromkeys(zeros, -0.0)
        for store, rotations in (
            (reference, None), (explicit_zero, zeros), (explicit_negative_zero, minus), (empty, {}),
        ):
            store[edge.key] = _wedges_at_half_edge(reference_mesh, edge, rotations, geometric)

    assert reference
    for label, store in (
        ("explicit 0.0", explicit_zero), ("explicit -0.0", explicit_negative_zero), ("empty map", empty),
    ):
        for key, wedges in reference.items():
            assert store[key] == wedges, f"{label} changed the wedges at half-edge {key}"


def test_a_geometrically_flat_interface_reads_at_the_numerical_floor():
    """A flat film's *fitted* curvature is zero only to floating point, and that floor is pinned.

    Built as a phantom so the true curvature is exactly zero by construction: three planar
    half-planes meeting along one straight junction line. The least-squares fit does **not**
    return an exactly-zero `c_ee` on it — the design matrix is not exactly rank-consistent in
    binary floating point — so this is the honest statement of what "identity at zero curvature"
    buys in practice, as opposed to the bitwise statement above.

    MEASURED on this phantom: max |rotation| 4.1e-17 rad = 2.3e-15 degrees, max change in a
    reported wedge angle 2.8e-14 degrees. Sixteen orders of magnitude below the 5-degree
    per-line errors the reading exists to reduce, and below `float64`'s own resolution at 120
    degrees. The bound below is loose on purpose: it is a floor, not a target, and it should not
    fail on a different BLAS.
    """
    mesh, _ = _flat_three_film_phantom()
    rotations, report = quadric_angle_rotations(mesh)
    assert report.coverage_fraction == 1.0, "the flat phantom must be fully covered"
    assert max(abs(angle) for angle in rotations.values()) < 1e-12

    chord_rad, chord_deg, chord_len = compute_angles_tri(mesh, unique=False)
    quadric_rad, quadric_deg, quadric_len, _report = compute_angles_tri_with_report(
        mesh, unique=False, reading=QUADRIC_READING,
    )
    assert set(chord_rad) == set(quadric_rad)
    assert all(chord_len[key] == quadric_len[key] for key in chord_len), "no reading may move a length"
    assert max(abs(chord_deg[key] - quadric_deg[key]) for key in chord_deg) < 1e-10


def test_safeguard_2_the_fit_is_generic_and_recovers_a_known_curvature():
    """A cylindrical film of known curvature must be read back at that curvature.

    The generic quadric is fitted to a discretised cylinder of radius `R` whose axis is
    perpendicular to the junction; the section curvature it reports at the junction must be
    `1/R` in magnitude. Nothing in the fit knows `R`, and nothing constrains the surface to
    constant mean curvature — that is what "generic" means and what this checks.
    """
    radius = 25.0
    mesh, _ = _one_curved_film_phantom(radius=radius)
    _rotations, report = quadric_angle_rotations(mesh)
    assert report.n_corrected > 0

    from foambryo.second_order_angle_reading import _fit_quadric_height_field, _section_curvature

    frames = junction_frames(mesh)
    curved = [f for f in frames if f["pair"] == (1, 2)]
    assert curved, "the phantom must carry the curved interface"
    curvatures = []
    for frame in curved:
        lever = frame["lever"]
        window = QUADRIC_WINDOW_FACTOR * lever
        from foambryo.second_order_angle_reading import _interface_vertices, _local_coordinates

        keys, coordinates, tree = _interface_vertices(mesh)[frame["pair"]]
        found = np.asarray(
            tree.query_ball_point(frame["midpoint"], window * np.hypot(1.0, QUADRIC_AXIAL_FACTOR) + 2.0),
            dtype=np.int64,
        )
        samples = coordinates[found]
        xi, eta, zeta = _local_coordinates(samples, frame)
        # The same keep rule the reader uses, including the explicit membership of the frame's own
        # two junction-edge endpoints -- so this test reads the fit the reader actually performs.
        is_junction_endpoint = np.isin(keys[found], np.asarray(frame["edge"], dtype=keys.dtype))
        keep = (
            (np.abs(xi) <= QUADRIC_AXIAL_FACTOR * window)
            & (np.hypot(eta, zeta) <= window)
            & ((eta > 0.0) | is_junction_endpoint)
        )
        fit = _fit_quadric_height_field(xi[keep], eta[keep], zeta[keep])
        if fit is not None:
            curvatures.append(abs(_section_curvature(fit)))
    assert curvatures
    assert np.median(curvatures) == pytest.approx(1.0 / radius, rel=0.15)


def test_safeguard_5_no_reading_path_can_change_the_window():
    """No route from a `reading=` selector, or from any inference entry point, to the window.

    The window multipliers are module constants at the curvature-from-data work's a-priori
    values, re-selected onto themselves by a later stage of the identifiability-floor work. A
    measurement may name `quadric_angle_rotations` and pass a window explicitly — that is how the
    cost of the window is measured without a second implementation of the fit — but **every path
    that a `reading=` selector reaches must be unable to**, so no benchmark, inference or public
    convenience call can tune the window against the quantity it is judged by.
    """
    import inspect

    assert QUADRIC_WINDOW_FACTOR == 4.0
    assert QUADRIC_AXIAL_FACTOR == 1.0
    for function in (compute_angles_tri, compute_angles_tri_with_report, compute_trijunction_edges,
                     compute_trijunction_edges_with_report, rotations_for_reading,
                     infer_tensions, infer_tensions_and_residuals_with_report,
                     infer_tensions_young_dupre_local):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"window_factor", "axial_factor", "collar_voxels"}


def test_the_window_parameters_are_the_identity_at_their_defaults(reference_mesh):
    """Executed, not short-circuited: the explicit call must reproduce the default one bitwise.

    The window multipliers were added so the reader's own floor could be measured against window
    size. The obligation that comes with them is that passing the shipped values explicitly is
    *the same computation*, key for key and float for float — otherwise a sweep's value at `4.0`
    would not be the number production reads and the sweep would adjudicate nothing.
    """
    default_rotations, default_report = quadric_angle_rotations(reference_mesh)
    explicit_rotations, explicit_report = quadric_angle_rotations(
        reference_mesh, window_factor=QUADRIC_WINDOW_FACTOR, axial_factor=QUADRIC_AXIAL_FACTOR,
    )
    assert default_rotations.keys() == explicit_rotations.keys()
    assert default_rotations  # the fixture mesh must actually exercise the fit
    for key, value in default_rotations.items():
        assert explicit_rotations[key] == value  # bitwise, not approx
    assert explicit_report == default_report
    assert default_report.window_factor == QUADRIC_WINDOW_FACTOR
    assert default_report.axial_factor == QUADRIC_AXIAL_FACTOR


def test_a_smaller_window_changes_the_reading_and_is_reported(reference_mesh):
    """A window that is not the default must produce a different map and say so in the report.

    The other half of the identity test: if the parameters were silently ignored, the identity
    above would pass vacuously.
    """
    rotations, report = quadric_angle_rotations(reference_mesh, window_factor=2.0)
    default_rotations, _default_report = quadric_angle_rotations(reference_mesh)
    assert report.window_factor == 2.0
    assert report.axial_factor == QUADRIC_AXIAL_FACTOR
    shared = rotations.keys() & default_rotations.keys()
    assert shared
    assert any(rotations[key] != default_rotations[key] for key in shared)


def test_the_null_control_degrades_the_reading():
    """Safeguard 3: a correction built from shuffled curvature must not help.

    Shuffling assigns every `(edge, face)` another pair's fitted rotation. The shuffled angles
    must differ from the real corrected ones — if a permutation of the curvature field were as
    good as the field itself, the correction would be an artefact of the pipeline rather than of
    the geometry. The *scientific* comparison against ground truth is the
    second-order-angle-reading work's job; this test only pins that the null is a genuinely
    different reading, so that comparison's null is not silently the same arm.
    """
    mesh, _ = _one_curved_film_phantom(radius=25.0)
    rotations, _report = quadric_angle_rotations(mesh)
    nonzero = {key: value for key, value in rotations.items() if value != 0.0}
    assert len(nonzero) > 1, "the phantom must carry a curved interface for a shuffle to matter"

    keys = sorted(rotations)
    values = [rotations[key] for key in keys]
    shuffled = dict(zip(keys, values[1:] + values[:1], strict=True))
    corrected = compute_angles_tri_with_report(mesh, unique=False, reading=QUADRIC_READING)[0]

    from foambryo.geometry import _wedges_at_half_edge

    def read(rotation_map):
        angles = {}
        for edge in mesh.half_edges:
            if len(edge.twin) <= 1:
                continue
            geometric = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
            wedges = _wedges_at_half_edge(mesh, edge, rotation_map, geometric)
            if wedges is None:
                continue
            for key, angle in wedges:
                angles.setdefault(key, []).append(angle)
        return {key: float(np.mean(value)) for key, value in angles.items()}

    null = read(shuffled)
    assert set(null) == set(corrected)
    assert any(null[key] != corrected[key] for key in null)


# ---------------------------------------------------------------------------------------
# 3 -- the fallback must be visible
# ---------------------------------------------------------------------------------------
def test_a_partly_covered_mesh_reports_its_uncorrected_lines():
    """A mesh too coarse for the quadric window must name the lines it could not correct.

    The phantom is deliberately built with only two rings of triangles per film, which leaves
    fewer than two mesh vertices per fit parameter in most windows. The report must show the
    shortfall and list the affected trijunction lines, and `is_complete` must be False.
    """
    mesh, _ = _under_covered_phantom()
    _rotations, report = quadric_angle_rotations(mesh)
    assert report.reading == QUADRIC_READING
    assert report.n_incident_faces > 0
    assert report.n_corrected < report.n_incident_faces, "this phantom is meant to be under-covered"
    assert 0.0 <= report.coverage_fraction < 1.0
    assert report.uncorrected_lines
    assert not report.is_complete
    for line in report.uncorrected_lines:
        corrected, total = report.line_coverage[line]
        assert corrected < total
    assert "fall back to the chord reading" in report.describe()


def test_the_coverage_warning_leads_with_its_numbers():
    """The warning fires on essentially every real mesh, so its message must carry the numbers.

    Measured on 199 meshes (154 real embryo timepoints and 45 benchmark cases), the warning fired on
    every one; firing alone therefore tells a user nothing. The message must state the measured
    face-coverage fraction and how many trijunction lines were not fully corrected, out of how many.
    """
    mesh, _ = _under_covered_phantom()
    _rotations, report = quadric_angle_rotations(mesh)
    message = report.describe()
    assert f"measured coverage {report.coverage_fraction:.4f}" in message
    counts = f"{len(report.uncorrected_lines)} of {len(report.line_coverage)} trijunction lines not fully corrected"
    assert counts in message
    with pytest.warns(AngleReadingCoverageWarning, match="measured coverage"):
        infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading=QUADRIC_READING)


def test_line_coverage_fraction_is_the_fraction_of_fully_covered_lines():
    """The line analogue of `coverage_fraction`: whole lines, not individual faces.

    A line with 2 of 3 faces corrected counts as 0 towards `line_coverage_fraction` (it is
    in `uncorrected_lines`) but as 2/3 towards the per-face `coverage_fraction` -- the two
    numbers must therefore not agree in general, and `line_coverage_fraction` must equal a
    hand count over `report.line_coverage`.
    """
    mesh, _ = _under_covered_phantom()
    _rotations, report = quadric_angle_rotations(mesh)

    expected = sum(1 for corrected, total in report.line_coverage.values() if corrected == total) / len(
        report.line_coverage,
    )
    assert report.line_coverage_fraction == pytest.approx(expected)
    assert 0.0 <= report.line_coverage_fraction < 1.0
    # Under-covered by construction, so at least one line is not fully covered -- the two
    # coverage notions must actually differ here, not coincide by accident.
    assert report.line_coverage_fraction != pytest.approx(report.coverage_fraction)


def test_line_coverage_fraction_is_none_on_a_mesh_with_no_trijunction_line():
    """Mirrors `coverage_fraction`'s own empty-mesh convention (`None`, not `0.0` or `nan`)."""
    empty_report = AngleReadingReport(
        reading=QUADRIC_READING,
        n_incident_faces=0,
        n_corrected=0,
        n_no_samples=0,
        n_no_fit=0,
        coverage_fraction=None,
        uncorrected_lines=(),
        line_coverage={},
    )
    assert empty_report.line_coverage_fraction is None


def test_the_report_less_paths_warn_when_the_report_less_paths_would_hide_it():
    """`compute_angles_tri` and `infer_tensions` must not mix conventions silently."""
    mesh, _ = _under_covered_phantom()
    with pytest.warns(AngleReadingCoverageWarning, match="fall back to the chord reading"):
        compute_angles_tri(mesh, unique=False, reading=QUADRIC_READING)
    with pytest.warns(AngleReadingCoverageWarning):
        compute_trijunction_edges(mesh, reading=QUADRIC_READING)
    with pytest.warns(AngleReadingCoverageWarning):
        infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading=QUADRIC_READING)


def test_the_report_returning_paths_hand_back_the_numbers_instead_of_a_warning():
    """A caller holding the coverage report does not also need to be shouted at."""
    mesh, _ = _under_covered_phantom()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AngleReadingCoverageWarning)
        _rad, _deg, _len, report = compute_angles_tri_with_report(mesh, unique=False, reading=QUADRIC_READING)
        assert not report.is_complete
        _edges, edge_report = compute_trijunction_edges_with_report(mesh, reading=QUADRIC_READING)
        assert not edge_report.is_complete
        *_values, inference_report = infer_tensions_and_residuals_with_report(
            mesh, 1, TensionComputationMethod.YoungDupre, angle_reading=QUADRIC_READING,
        )
        assert not inference_report.is_complete
        # The quadric-safeguard work: YoungDupreLocal refuses the quadric reading outright rather than accepting an
        # incomplete-coverage report for it — there is no report-returning path here any more.
        with pytest.raises(ValueError, match="quadric"):
            infer_tensions_young_dupre_local(mesh, 1.0, angle_reading=QUADRIC_READING)


def test_the_chord_reading_reports_complete_coverage(reference_mesh):
    """The chord reading has no fit that can fail, so it never falls back and never warns."""
    _rad, _deg, _len, report = compute_angles_tri_with_report(reference_mesh, unique=False)
    assert report.reading == CHORD_READING
    assert report.coverage_fraction == 1.0
    assert report.is_complete
    assert report.uncorrected_lines == ()


# ---------------------------------------------------------------------------------------
# 4 -- the reading actually reaches every method that claims to honour it
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("method_name", ANGLE_METHOD_NAMES)
def test_every_angle_reading_method_actually_changes_under_the_quadric_reading(method_name):
    """If a method accepts `angle_reading` its answer must depend on it — otherwise the API lies.

    Run on a phantom with one genuinely curved film, where the correction is non-zero, so a
    method that quietly ignored the selector would return identical tensions and fail here.
    """
    mesh, _ = _one_curved_film_phantom(radius=25.0)
    mode = getattr(TensionComputationMethod, method_name)
    if method_name == "YoungDupreLocal":
        # The quadric-safeguard work: this combination is refused outright, not silently honoured — see
        # test_young_dupre_local_refuses_the_quadric_reading.
        with pytest.raises(ValueError, match="quadric"):
            infer_tensions(mesh, 1, mode, angle_reading=QUADRIC_READING)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AngleReadingCoverageWarning)
        chord = infer_tensions(mesh, 1, mode, angle_reading=CHORD_READING)
        quadric = infer_tensions(mesh, 1, mode, angle_reading=QUADRIC_READING)
    assert set(chord) == set(quadric)
    assert any(chord[key] != quadric[key] for key in chord), f"{method_name} ignored angle_reading"


def test_the_rotation_map_is_keyed_by_productions_own_trijunctions(reference_mesh):
    """Every rotation key must name a geometric edge production actually visits, and one of its faces.

    This is the identity requirement of an early stage of the second-order-angle-reading work,
    reduced to something a unit test can hold:
    a rotation keyed to an edge or a face production does not read would be applied to nothing,
    and a rotation keyed to the *wrong* face would corrupt an angle silently.
    """
    rotations, _report = quadric_angle_rotations(reference_mesh)
    production_edges = {
        (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
        for edge in reference_mesh.half_edges
        if len(edge.twin) > 1
    }
    for (geometric_edge, face_key), angle in rotations.items():
        assert geometric_edge in production_edges
        assert 0 <= face_key < len(reference_mesh.f)
        assert np.isfinite(angle)
        triangle = {int(v) for v in reference_mesh.f[face_key, :3]}
        assert set(geometric_edge) < triangle, "the rotated face must be incident on its own edge"


# ---------------------------------------------------------------------------------------
# phantoms
# ---------------------------------------------------------------------------------------
def _under_covered_phantom():
    """A junction the quadric can correct on two of its three films but not on the third.

    Films `(0, 1)` and `(0, 2)` carry six rings; film `(1, 2)` carries one, on a five-vertex
    junction, which leaves five interface vertices in its window against the ten the fit needs.
    Its faces therefore fall back to the chord reading, and the single trijunction line
    `(0, 1, 2)` is partly corrected — exactly the mixed state the report exists to surface.

    Returns:
        tuple[DcelData, dict]: the mesh and the three films' material pairs.
    """
    return _flat_three_film_phantom(rings_per_film=(6, 6, 1), n_axis=5)


def _incident_faces(mesh, edge):
    """The three faces `_wedges_at_half_edge` collects at one half-edge, in its own order."""
    return [edge.incident_face, *(mesh.half_edges[twin].incident_face for twin in edge.twin)]


def _flat_three_film_phantom(rings: int = 6, rings_per_film=None, n_axis: int = 9):
    """Three planar films meeting at 120 degrees along one straight junction line.

    Exactly zero curvature everywhere, so the quadric reading has a known answer. `rings` (or
    `rings_per_film`, per film) controls how many triangle rings each film carries away from the
    junction; `n_axis` how many vertices the junction line itself has. Starving one film of both
    is how the deliberately under-covered variant is built: the quadric needs two mesh vertices
    per fit parameter — ten — inside its window, and a one-ring film on a five-vertex junction
    offers five.

    Returns:
        tuple[DcelData, dict]: the mesh and the three films' material pairs.
    """
    rings_per_film = (rings, rings, rings) if rings_per_film is None else tuple(rings_per_film)
    axis_positions = np.linspace(0.0, 40.0, n_axis)
    directions = [
        np.array([np.cos(theta), np.sin(theta)]) for theta in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]
    spacing = 6.0
    junction = np.stack([np.zeros(n_axis), np.zeros(n_axis), axis_positions], axis=1)
    points = [junction]
    offset = n_axis
    film_columns = []
    for direction, film_rings in zip(directions, rings_per_film, strict=True):
        columns = []
        for ring in range(1, film_rings + 1):
            radius = ring * spacing
            block = np.stack(
                [np.full(n_axis, radius * direction[0]), np.full(n_axis, radius * direction[1]), axis_positions],
                axis=1,
            )
            points.append(block)
            columns.append(np.arange(offset, offset + n_axis))
            offset += n_axis
        film_columns.append(columns)
    points = np.concatenate(points, axis=0)

    # films: (0,1), (0,2), (1,2) with 0 the exterior, so every film separates two labels and the
    # three of them share the junction.
    pairs = [(0, 1), (0, 2), (1, 2)]
    triangles, labels = [], []
    for columns, pair in zip(film_columns, pairs, strict=True):
        strips = [np.arange(n_axis), *columns]
        for left, right in itertools.pairwise(strips):
            for i in range(n_axis - 1):
                triangles.append([left[i], right[i], left[i + 1]])
                labels.append(list(pair))
                triangles.append([right[i], right[i + 1], left[i + 1]])
                labels.append(list(pair))
    mesh = DcelData(points, np.asarray(triangles, dtype=np.int64), np.asarray(labels, dtype=np.int64))
    return mesh, {"pairs": pairs}


def _one_curved_film_phantom(radius: float = 25.0, rings: int = 6):
    """The same three-film junction, with the `(1, 2)` film bent onto a cylinder of `radius`.

    The bent film's section in the plane normal to the junction is a circular arc of curvature
    `1/radius`, so the quadric fit has a known answer and the correction is non-zero.

    Returns:
        tuple[DcelData, dict]: the mesh and its material pairs.
    """
    mesh, meta = _flat_three_film_phantom(rings=rings)
    points = mesh.v.copy()
    on_curved = np.zeros(len(points), dtype=bool)
    for row in mesh.f:
        if (int(row[3]), int(row[4])) == (1, 2):
            on_curved[row[:3]] = True
    # Bend along the film's own outward direction: the (1, 2) film runs along `directions[2]`.
    direction = np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)])
    normal = np.array([-direction[1], direction[0]])
    for index in np.flatnonzero(on_curved):
        along = float(points[index, :2] @ direction)
        if along <= 0:
            continue
        points[index, :2] = along * direction + (radius - np.sqrt(max(radius**2 - along**2, 0.0))) * normal
    bent = DcelData(points, mesh.f[:, :3].copy(), mesh.f[:, 3:5].copy())
    return bent, meta


# ---------------------------------------------------------------------------------------
# 12 -- the frame a mesh is expressed in
# ---------------------------------------------------------------------------------------
#: What a reading's line-mean wedge angles are allowed to move by when the identical mesh is
#: rescaled. An angle between two face normals does not depend on the units the mesh is measured in,
#: so the only admissible movement is floating-point rounding. **Both** readings now meet this. The
#: measured worst cases on the fixture are `2.5e-13` degrees for the chord reading and `2.3e-13` for
#: the second-order one, and this bound is four orders above both.
MAX_SCALE_SENSITIVITY_DEG = 1e-9
#: Kept under its old name so that anything importing it still resolves. Both readings share one
#: bound now; there is no longer a second, looser one for the second-order reading.
MAX_CHORD_SCALE_SENSITIVITY_DEG = MAX_SCALE_SENSITIVITY_DEG
#: Scale factors the sensitivity is checked at. The benchmark's reference meshes are stored in
#: normalised units whose bounding box spans about `1.36`, while the reconstructions of the same
#: cases are in voxels and span about `159`, so the two mesh families of one dataset differ by
#: roughly this much and the question is not academic: the same reference meshes read `2.032`
#: degrees of contact-angle error in their stored frame and `2.046` in the voxel frame, and both
#: numbers were published under the same name.
SCALE_FACTORS = (10.0, 117.0, 1000.0)


def _rescaled(mesh, scale: float) -> DcelData:
    """The same mesh with every coordinate multiplied by `scale`; connectivity untouched."""
    triangles_and_labels = np.asarray(mesh.f, dtype=np.int64)
    return DcelData(
        np.asarray(mesh.v, dtype=np.float64) * scale,
        triangles_and_labels[:, :3],
        triangles_and_labels[:, 3:5],
    )


def _worst_shift_deg(mesh, reading: str, scale: float) -> float:
    """Largest line-mean wedge-angle movement, in degrees, when the mesh is rescaled."""
    baseline_angles, _deg, _lengths = compute_angles_tri(mesh, unique=False, reading=reading)
    angles, _deg, _lengths = compute_angles_tri(_rescaled(mesh, scale), unique=False, reading=reading)
    shared = set(angles) & set(baseline_angles)
    assert shared, "the two readings share no wedge key, which is a different failure"
    return max(abs(float(np.degrees(angles[key] - baseline_angles[key]))) for key in shared)


@pytest.mark.parametrize("reading", [CHORD_READING, QUADRIC_READING])
def test_both_readings_are_invariant_under_a_change_of_units(reference_mesh, reading):
    """An angle between two surfaces cannot depend on the units the surfaces are measured in.

    Invariant **to rounding**, not bitwise: the normals, the fit and `arccos` are each evaluated in
    floating point and a scaled mesh takes a different rounding path, so some wedge keys do move --
    in the thirteenth decimal place.

    **This test replaces one that asserted the opposite**, and the replacement is deliberate. The
    second-order reading used not to have this invariance: its fit kept a sample when `eta > 0.0`
    strictly, and the frame's own two junction-edge endpoints have `eta` exactly zero in exact
    arithmetic, so their membership was decided by the sign of a rounding error -- which changes with
    scale. Rescaling this fixture by `117x` changed the kept-sample count on `652` of `1 065`
    incident faces and moved the line-mean contact angle by up to `0.14` degrees, and that is why
    `2.032` and `2.046` were both published for one row. The predecessor test bounded the defect at
    `0.25` degrees so that it could not silently get worse, and said in its own docstring that
    whoever repaired it would have to remove it deliberately. It is removed deliberately here.

    The repair decides those two samples by **vertex identity** instead of by a computed zero;
    `second_order_angle_reading.JUNCTION_ENDPOINTS` names the policy. Measured after it, on this
    fixture: `2.3e-13` degrees for the second-order reading against `2.5e-13` for the chord reading,
    a factor of about `6e11` better than before and now the same rounding level as the reading that
    always had the property.
    """
    for scale in SCALE_FACTORS:
        worst = _worst_shift_deg(reference_mesh, reading, scale)
        assert worst <= MAX_SCALE_SENSITIVITY_DEG, (
            f"the {reading} reading moved by {worst:.3e} degrees when the mesh was scaled by "
            f"{scale}, above the rounding-level bound of {MAX_SCALE_SENSITIVITY_DEG:.0e}"
        )


def test_the_junction_endpoint_policy_is_what_makes_the_reading_invariant(reference_mesh):
    """Executed rather than asserted: the historical policy still reproduces the old defect.

    The repair is only meaningful if the thing it repaired is still demonstrably there when it is
    switched off. `junction_endpoints="rounding"` keeps the historical `eta > 0.0` test, and under
    it the reading must still move by far more than the rounding-level bound -- otherwise the
    invariance above would be evidence about something else.
    """
    from foambryo.second_order_angle_reading import quadric_angle_rotations as rotations_of

    def worst_under(policy: str) -> float:
        worst = 0.0
        for scale in SCALE_FACTORS:
            angles = {}
            for name, candidate in (("base", reference_mesh), ("scaled", _rescaled(reference_mesh, scale))):
                candidate._angle_reading_rotations = rotations_of(candidate, junction_endpoints=policy)
                angles[name], _deg, _lengths = compute_angles_tri(
                    candidate, unique=False, reading=QUADRIC_READING,
                )
            shared = set(angles["base"]) & set(angles["scaled"])
            worst = max(worst, *(abs(float(np.degrees(angles["scaled"][k] - angles["base"][k]))) for k in shared))
        return worst

    historical = worst_under("rounding")
    repaired = worst_under("include")
    assert historical > 1e-4, (
        f"the historical policy moved by only {historical:.3e} degrees, so this fixture no longer "
        f"exhibits the defect the repair exists for and the invariance test above proves less than "
        f"it appears to"
    )
    assert repaired <= MAX_SCALE_SENSITIVITY_DEG
    assert historical > 1e6 * repaired
