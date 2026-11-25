"""The circular-arc reading and the 2D pressures of `foambryo._two_dimensional`.

Pinned: on exact equilibrium geometries built here (a symmetric doublet with a straight shared interface, and a
double bubble of unequal cells, all tensions equal, whose shared interface is curved), the arc reading returns the
exact tensions and `infer_pressures` the exact Laplace pressures, signs included; an interface with fewer than 4
vertices is read by its chord and gives no pressure row; cells linked to the medium by no row get `nan`, never a
number; on the 18 golden meshes every interface's edges agree on which material is on their left (dw2d's
orientation contract), and a mesh that breaks it is warned about; `"arc"` is the default; the component entry
point returns `ComponentTensionSolution` without warning; `diagnostics` returns the report.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import ComponentTensionSolution, JunctionGraphGaugeWarning
from foambryo.linear_solves import SolveDiagnostics

DATA = Path(__file__).resolve().parent / "data"
SPACING_PX = 2.0


def _arc_points(centre: np.ndarray, radius: float, start: float, stop: float) -> np.ndarray:
    """Points of a circle from angle `start` to `stop` (radians, counter-clockwise if stop > start), (x, y)."""
    n = max(4, int(np.ceil(abs(stop - start) * radius / SPACING_PX)) + 1)
    angles = np.linspace(start, stop, n)
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _mesh_from_polylines(polylines: list[tuple[np.ndarray, int, int]], label_of) -> tuple[np.ndarray, np.ndarray]:
    """A (row, col) mesh from (x, y) polylines `(points, a, b)` sharing their end points, oriented as dw2d's.

    Every edge gets `m1 < m2` and `m1` on the left of `v1 -> v2` in (x, y), decided by `label_of(point)`.
    """
    index: dict[tuple[float, float], int] = {}
    vertices, edges = [], []

    def vertex(p: np.ndarray) -> int:
        key = (round(float(p[0]), 9), round(float(p[1]), 9))
        if key not in index:
            index[key] = len(vertices)
            vertices.append([p[1], p[0]])
        return index[key]

    for points, a, b in polylines:
        ids = [vertex(p) for p in points]
        for k in range(len(ids) - 1):
            i, j, p, q = ids[k], ids[k + 1], points[k], points[k + 1]
            d = q - p
            # 0.2 px off the chord's midpoint: well beyond the sagitta (about 0.02 px here), well inside the region
            left = 0.5 * (p + q) + 0.2 * np.array([-d[1], d[0]]) / np.hypot(*d)
            m1, m2 = min(a, b), max(a, b)
            edges.append([i, j, m1, m2] if label_of(left) == m1 else [j, i, m1, m2])
    return np.array(vertices, float), np.array(edges, np.int64)


# The synthetic meshes below are built in the former orientation (dw2d before 0.8.0: `m1` on the left of each edge
# in `(x, y) = (column, row)`), which foambryo measures and reverses on input
# (`test_two_dimensional_orientation_on_input.py`).
def _doublet(radius: float = 40.0, offset: float = 20.0):
    """Two cells of radius R whose centres are `offset` from a straight shared interface; outer tensions 1.

    Young-Dupre at a junction: gamma_12 = 2 offset / R; Laplace: p_1 = p_2 = 1 / R (medium 0).
    """
    h = np.sqrt(radius**2 - offset**2)
    c1, c2 = np.array([-offset, 0.0]), np.array([offset, 0.0])
    beta = np.arctan2(h, offset)  # angle of the top junction seen from the left centre
    outer_1 = _arc_points(c1, radius, beta, 2 * np.pi - beta)  # around the left, top to bottom
    outer_2 = _arc_points(c2, radius, np.pi - beta, -np.pi + beta)  # around the right, top to bottom
    shared = np.column_stack([np.zeros(9), np.linspace(h, -h, 9)])

    def label_of(p: np.ndarray) -> int:
        if p[0] < 0 and np.hypot(*(p - c1)) < radius:
            return 1
        if p[0] > 0 and np.hypot(*(p - c2)) < radius:
            return 2
        return 0

    mesh = _mesh_from_polylines([(outer_1, 0, 1), (outer_2, 0, 2), (shared, 1, 2)], label_of)
    tensions = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 2 * offset / radius}
    return mesh, tensions, {0: 0.0, 1: 1 / radius, 2: 1 / radius}


def _double_bubble(r1: float = 30.0, r2: float = 45.0):
    """The standard double bubble: all tensions 1, angles of 120 degrees, 1 / R_12 = 1 / R_1 - 1 / R_2.

    The centres are D apart with D^2 = R_1^2 + R_2^2 - R_1 R_2 (the radii to a junction make 60 degrees); the
    shared interface is the circle through both junctions with radius R_1 R_2 / (R_2 - R_1), bulging into the
    larger cell. Laplace: p_1 = 1 / R_1, p_2 = 1 / R_2.
    """
    distance = np.sqrt(r1**2 + r2**2 - r1 * r2)
    x1 = (distance**2 + r1**2 - r2**2) / (2 * distance)  # the junctions' abscissa from the centre of cell 1
    h = np.sqrt(r1**2 - x1**2)
    c1, c2 = np.array([-x1, 0.0]), np.array([distance - x1, 0.0])
    r12 = r1 * r2 / (r2 - r1)
    c12 = np.array([-np.sqrt(r12**2 - h**2), 0.0])  # on the side of the smaller cell: bulges into cell 2
    a1 = np.arctan2(h, x1)
    a2 = np.arctan2(h, -(distance - x1))
    a12 = np.arctan2(h, -c12[0])
    outer_1 = _arc_points(c1, r1, a1, 2 * np.pi - a1)
    outer_2 = _arc_points(c2, r2, a2, -a2)
    shared = _arc_points(c12, r12, a12, -a12)

    def label_of(p: np.ndarray) -> int:
        inside_1, inside_2 = np.hypot(*(p - c1)) < r1, np.hypot(*(p - c2)) < r2
        if inside_1 and np.hypot(*(p - c12)) < r12:
            return 1
        if inside_2 or inside_1:
            return 2
        return 0

    mesh = _mesh_from_polylines([(outer_1, 0, 1), (outer_2, 0, 2), (shared, 1, 2)], label_of)
    return mesh, {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}, {0: 0.0, 1: 1 / r1, 2: 1 / r2}


@pytest.mark.parametrize("case", [_doublet, _double_bubble], ids=["doublet", "double_bubble"])
def test_exact_arcs_give_exact_tensions_and_pressures(case):
    mesh, true_tensions, true_pressures = case()
    tensions = fb2.infer_tensions(mesh, exclude_image_border=False)
    scale = np.mean(list(true_tensions.values())) / np.mean([tensions[k] for k in true_tensions])
    for key, value in true_tensions.items():
        assert tensions[key] * scale == pytest.approx(value, rel=1e-9), key
    pressures = fb2.infer_pressures(mesh, {k: v * scale for k, v in tensions.items()}, exclude_image_border=False)
    for material, value in true_pressures.items():
        assert pressures[material] == pytest.approx(value, rel=1e-8, abs=1e-12), material


def test_the_chord_reading_is_not_exact_on_the_same_arcs():
    """The control: the chord reads the first edge, not the arc's tangent, so it misses the doublet's tensions."""
    mesh, true_tensions, _ = _double_bubble()
    tensions = fb2.infer_tensions(mesh, angle_reading="chord", exclude_image_border=False)
    assert max(abs(tensions[k] - v) for k, v in true_tensions.items()) > 1e-3


def test_the_arcs_of_the_double_bubble():
    mesh, _, _ = _double_bubble()
    arcs = {arc.materials: arc for arc in fb2.interface_arcs(mesh, exclude_image_border=False)}
    assert set(arcs) == {(0, 1), (0, 2), (1, 2)}
    assert all(arc.fit == "circle" and arc.n_edges_against_the_majority == 0 for arc in arcs.values())
    # the sign follows the traversal: "left" is the material on the left of vertices[0] -> vertices[-1]

    def toward(arc, material):  # the curvature with the circle's centre counted positive inside `material`
        return arc.curvature if arc.left == material else -arc.curvature

    # each cell bulges into the medium, and the smaller cell (higher pressure) into the larger one
    assert toward(arcs[(0, 1)], 1) == pytest.approx(1 / 30.0, rel=1e-9)
    assert toward(arcs[(0, 2)], 2) == pytest.approx(1 / 45.0, rel=1e-9)
    assert toward(arcs[(1, 2)], 1) == pytest.approx(1 / 30.0 - 1 / 45.0, rel=1e-9)


def test_a_short_interface_is_read_by_its_chord_and_gives_no_pressure_row():
    """Three straight interfaces of 3 vertices from one junction: chord fallback, no curvature, no pressure."""
    vertices = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0], [4.0, -3.0], [8.0, -6.0], [-4.0, -3.0], [-8.0, -6.0]])
    edges = np.array([[0, 1, 1, 2], [1, 2, 1, 2], [0, 3, 2, 3], [3, 4, 2, 3], [0, 5, 1, 3], [5, 6, 1, 3]])
    arcs = fb2.interface_arcs((vertices, edges), exclude_image_border=False)
    assert len(arcs) == 3
    assert all(arc.fit == "chord" for arc in arcs)
    assert all(np.isnan(arc.curvature) for arc in arcs)
    tensions = fb2.infer_tensions((vertices, edges), exclude_image_border=False)
    chord = fb2.infer_tensions((vertices, edges), angle_reading="chord", exclude_image_border=False)
    assert tensions == pytest.approx(chord, rel=1e-12)  # straight interfaces: the whole chord is the first edge
    pressures, report = fb2.infer_pressures(
        (vertices, edges),
        tensions,
        exclude_image_border=False,
        diagnostics=True,
    )
    assert report.n_rows == 0
    assert all(np.isnan(pressures[m]) for m in (1, 2, 3))


def test_cells_not_linked_to_the_medium_are_undetermined_not_a_number():
    """Two cells linked to each other by one Laplace row but not to the medium: both pressures are nan.

    Their outer interfaces have 3 vertices (no curvature, no row); their shared interface is a curved arc.
    """
    top, bottom, left, right = (
        np.array([0.0, 30.0]),
        np.array([0.0, -30.0]),
        np.array([-40.0, 0.0]),
        np.array([40.0, 0]),
    )
    half_angle = np.arctan2(30.0, 60.0)
    shared = _arc_points(np.array([-60.0, 0.0]), np.hypot(60.0, 30.0), half_angle, -half_angle)

    def label_of(p: np.ndarray) -> int:
        if abs(p[0]) > 40 * (1 - abs(p[1]) / 30):
            return 0
        return 1 if np.hypot(*(p - np.array([-60.0, 0.0]))) < np.hypot(60.0, 30.0) else 2

    mesh = _mesh_from_polylines(
        [(np.array([top, left, bottom]), 0, 1), (np.array([top, right, bottom]), 0, 2), (shared, 1, 2)],
        label_of,
    )
    arcs = {arc.materials: arc for arc in fb2.interface_arcs(mesh, exclude_image_border=False)}
    assert arcs[(0, 1)].fit == arcs[(0, 2)].fit == "chord"
    assert arcs[(1, 2)].fit == "circle"
    tensions = fb2.infer_tensions(mesh, exclude_image_border=False)
    pressures, report = fb2.infer_pressures(mesh, tensions, exclude_image_border=False, diagnostics=True)
    assert (report.n_rows, report.n_cells_determined, report.n_cells_undetermined) == (0, 0, 2)
    assert pressures[0] == 0.0
    assert np.isnan(pressures[1])
    assert np.isnan(pressures[2])


@pytest.fixture(scope="module")
def golden_meshes():
    reference = json.loads((DATA / "two_dimensional_reference_tensions.json").read_text())
    with np.load(DATA / "two_dimensional_reference_meshes.npz") as arrays:
        return {n: (arrays[f"{n}__vertices"], arrays[f"{n}__edges"]) for n in sorted(reference["cases"])}


def test_every_interface_of_the_golden_meshes_follows_the_orientation_contract(golden_meshes):
    for name, mesh in golden_meshes.items():
        assert all(arc.n_edges_against_the_majority == 0 for arc in fb2.interface_arcs(mesh)), name


def test_a_mesh_that_breaks_the_orientation_contract_is_warned_about():
    """One edge reversed: the cells' boundaries still measure one orientation on input, the interface warns."""
    (vertices, edges), tensions, _ = _doublet()
    flipped = edges.copy()
    k = int(np.flatnonzero(edges[:, 3] == 1)[0])
    flipped[k, :2] = flipped[k, [1, 0]]
    with pytest.warns(UserWarning, match="does not follow the orientation rule on every edge"):
        pressures = fb2.infer_pressures((vertices, flipped), tensions, exclude_image_border=False)
    reference = fb2.infer_pressures((vertices, edges), tensions, exclude_image_border=False)
    assert pressures == pytest.approx(reference, rel=1e-12)


def test_the_default_reading_is_the_arc(golden_meshes):
    mesh = golden_meshes["mask_Cells_3__md3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        default, report = fb2.infer_tensions(mesh, diagnostics=True)
        arc = fb2.infer_tensions(mesh, angle_reading="arc")
        chord = fb2.infer_tensions(mesh, angle_reading="chord")
    assert fb2.DEFAULT_ANGLE_READING == "arc"
    assert report.angle_reading == "arc"
    assert all(default[k] == arc[k] or (np.isnan(default[k]) and np.isnan(arc[k])) for k in default)
    assert any(abs(default[k] - chord[k]) > 1e-6 for k in default if np.isfinite(default[k]))
    assert set(report.n_interfaces_by_fit) == {"circle", "line", "chord"}
    assert isinstance(report.solves, SolveDiagnostics)


def test_the_component_entry_point_returns_the_3d_solution_without_warning(golden_meshes):
    mesh = golden_meshes["mask_Cells_3__md3"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", JunctionGraphGaugeWarning)
        solution = fb2.infer_tensions_by_junction_component(mesh)
    assert isinstance(solution, ComponentTensionSolution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        flat = fb2.infer_tensions(mesh)
    assert all(solution.dict_tensions[k] == flat[k] or np.isnan(flat[k]) for k in flat)
    assert {k for k, c in solution.component_of.items() if c is None} == {k for k in flat if np.isnan(flat[k])}


def test_the_image_border_and_the_named_vertices_are_excluded_together(golden_meshes):
    mesh = golden_meshes["c_elegans__md3"]
    junctions, _ = fb2.junction_table(mesh, angle_reading="chord")
    inside = [j["vertex"] for j in junctions if j["vertex"] not in fb2.image_border_vertices(mesh)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        _, both = fb2.infer_tensions(mesh, excluded_vertices={inside[0]}, diagnostics=True)
        _, border_only = fb2.infer_tensions(mesh, diagnostics=True)
    assert both.n_junctions_excluded == border_only.n_junctions_excluded + 1


ARC_REFERENCE = json.loads((DATA / "two_dimensional_arc_reading_reference.json").read_text())


#: The reference was written by R188's research implementation, whose circles SciPy's Levenberg-Marquardt fit
#: (`least_squares(method="lm")`) stops where the sums of squared distances agree to rounding: on the flat minima of
#: few-point, shallow arcs that fixes a circle only to about the square root of the machine epsilon (two SciPy
#: releases differed by 8.0e-10 in tension on these meshes). foambryo's fit, written in NumPy, polishes the circle to
#: a stationary point and is the same on every NumPy and SciPy; it differs from the reference by up to 4.8e-9 of the
#: gauge in tension and 1.5e-9 of the largest pressure (measured on these 18 meshes, NumPy 2.0 to 2.5, SciPy 1.13 to
#: 1.18). The gate is 1e-8 on every version; `test_two_dimensional_circle_fit.py` pins the shipped reading to 1e-12.
ARC_REFERENCE_TOLERANCE = 1e-8


@pytest.mark.parametrize("name", sorted(ARC_REFERENCE["cases"]))
def test_the_arc_reading_is_that_of_its_research_implementation(golden_meshes, name):
    """The arc reading's tensions and pressures are its research implementation's, to `ARC_REFERENCE_TOLERANCE`.

    Against `two_dimensional_arc_reading_reference.json`, written by `make_two_dimensional_arc_reading_reference.py`.
    """
    reference = ARC_REFERENCE["cases"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        tensions, report = fb2.infer_tensions(golden_meshes[name], diagnostics=True)
    expected = {(a, b): (np.nan if v is None else v) for a, b, v in reference["tensions"]}
    assert set(tensions) == set(expected)
    for key, value in expected.items():
        assert np.isnan(tensions[key]) == np.isnan(value), key
        if np.isfinite(value):
            assert abs(tensions[key] - value) <= ARC_REFERENCE_TOLERANCE, key
    assert report.n_junctions_used == reference["n_junctions_used"]
    assert report.n_interfaces_by_fit["chord"] == reference["n_interfaces_read_by_their_chord"]
    with warnings.catch_warnings():  # the reference solves a fragmented mesh's rows together, as a plain map does
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        pressures = fb2.infer_pressures(golden_meshes[name], tensions)
    finite = [v for _, v in reference["pressures"] if v is not None]
    scale = max([abs(v) for v in finite] + [1e-300])
    for material, value in reference["pressures"]:
        assert np.isnan(pressures[material]) == (value is None), material
        if value is not None:
            assert abs(pressures[material] - value) <= ARC_REFERENCE_TOLERANCE * scale, material
