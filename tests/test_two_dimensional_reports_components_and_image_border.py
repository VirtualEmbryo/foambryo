"""The 2D reports, the pressures of each junction-graph component, and the image border read from dw2d's mark.

* `diagnostics=True` returns frozen, picklable reports (`TwoDimensionalReport`, `TwoDimensionalPressureReport`)
  that hold the 3D `SolveDiagnostics`.
* `infer_pressures` given a `ComponentTensionSolution` solves each component's Laplace rows alone: each
  component's pressures are exactly those of a plain solve given that component's tensions only, and a cell that
  rows of several components determine is `nan`, with a warning. Given a plain map whose rows span several
  components, it warns.
* A mesh that carries `image_border_vertices` (dw2d 0.8.0's `DCEL_Data`) is excluded by that mark, not by the
  vertex positions: a border junction that moved off the image rectangle is still excluded.
"""

import dataclasses
import pickle
import warnings

import numpy as np
import pytest

from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import ComponentTensionSolution, JunctionGraphGaugeWarning
from foambryo.linear_solves import SolveDiagnostics
from tests.test_two_dimensional_arc_reading import _double_bubble


def _two_double_bubbles():
    """Two double bubbles 300 px apart: cells 1, 2 and cells 3, 4, two junction-graph components."""
    (v1, e1), _, p1 = _double_bubble(30.0, 45.0)
    (v2, e2), _, p2 = _double_bubble(25.0, 50.0)
    v2 = v2 + np.array([0.0, 300.0])
    relabel = np.array([0, 3, 4])
    e2 = np.column_stack([e2[:, :2] + len(v1), relabel[e2[:, 2:]]])
    pressures = {0: 0.0, 1: p1[1], 2: p1[2], 3: p2[1], 4: p2[2]}
    return (np.vstack([v1, v2]), np.vstack([e1, e2])), pressures


def test_the_reports_are_frozen_picklable_and_hold_the_solve_diagnostics():
    mesh, _, _ = _double_bubble()
    solution, report = fb2.infer_tensions_by_junction_component(mesh, exclude_image_border=False, diagnostics=True)
    _, pressure_report = fb2.infer_pressures(mesh, solution, exclude_image_border=False, diagnostics=True)
    for r in (report, pressure_report):
        assert isinstance(r.solves, SolveDiagnostics)
        assert len(r.solves) >= 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.solves = None  # type: ignore[misc]
        again = pickle.loads(pickle.dumps(r))  # noqa: S301 -- our own object; a solve record holds arrays, so compare the scalars
        assert dataclasses.replace(again, solves=None) == dataclasses.replace(r, solves=None)
        assert again.solves.table() == r.solves.table()
    assert isinstance(report, fb2.TwoDimensionalReport)
    assert isinstance(pressure_report, fb2.TwoDimensionalPressureReport)
    assert (report.n_junctions_used, report.n_components) == (2, 1)
    assert report.image_border_from is None
    assert pressure_report.image_border_from is None
    assert pressure_report.per_component
    assert pressure_report.n_cells_determined == 2
    assert pressure_report.component_of_cell == {1: 0, 2: 0}


def test_each_component_is_solved_on_its_own_scale():
    mesh, true_pressures = _two_double_bubbles()
    solution = fb2.infer_tensions_by_junction_component(mesh, exclude_image_border=False)
    assert solution.n_components == 2
    pressures, report = fb2.infer_pressures(mesh, solution, exclude_image_border=False, diagnostics=True)
    assert report.n_cells_in_several_components == 0
    for component, cells in report.pressures_by_component.items():
        alone = {k: (t if solution.component_of[k] == component else np.nan) for k, t in solution.dict_tensions.items()}
        plain = fb2.infer_pressures(mesh, alone, exclude_image_border=False)
        assert {m: plain[m] for m in cells} == cells  # bitwise: the same rows, the same solve
    # every interface has tension 1 in truth and each component's gauge is a mean of 1: exact pressures
    for material, value in true_pressures.items():
        assert pressures[material] == pytest.approx(value, rel=1e-8, abs=1e-12), material


def test_a_plain_map_spanning_two_components_warns_and_one_component_does_not():
    mesh, _ = _two_double_bubbles()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        tensions = fb2.infer_tensions(mesh, exclude_image_border=False)
    with pytest.warns(JunctionGraphGaugeWarning, match="several components"):
        fb2.infer_pressures(mesh, tensions, exclude_image_border=False)
    single, _, _ = _double_bubble()
    with warnings.catch_warnings():
        warnings.simplefilter("error", JunctionGraphGaugeWarning)
        fb2.infer_pressures(single, fb2.infer_tensions(single, exclude_image_border=False), exclude_image_border=False)


def test_a_cell_determined_by_two_components_is_nan_and_each_component_keeps_its_pressure():
    """A hand-made partition of the double bubble: (0, 1) and (1, 2) in one component, (0, 2) in another."""
    mesh, _, true_pressures = _double_bubble(30.0, 45.0)
    keys = [(0, 1), (0, 2), (1, 2)]
    solution = ComponentTensionSolution(
        tensions=np.ones(3),
        dict_tensions=dict.fromkeys(keys, 1.0),
        component_of={(0, 1): 0, (1, 2): 0, (0, 2): 1},
        gauge_notes={0: "", 1: ""},
        undetermined=(),
        lstsq_residuals=np.zeros(2),
        fragmented=True,
        n_components=2,
    )
    with pytest.warns(JunctionGraphGaugeWarning, match="1 cell"):
        pressures, report = fb2.infer_pressures(mesh, solution, exclude_image_border=False, diagnostics=True)
    assert pressures[1] == pytest.approx(true_pressures[1], rel=1e-8)
    assert np.isnan(pressures[2])
    assert report.n_cells_in_several_components == 1
    assert report.component_of_cell == {1: 0, 2: None}
    assert report.pressures_by_component[0][2] == pytest.approx(true_pressures[2], rel=1e-8)
    assert report.pressures_by_component[1] == {2: pytest.approx(true_pressures[2], rel=1e-8)}


class _MarkedMesh:
    """A mesh object with dw2d 0.8.0's mark of the image border."""

    def __init__(self, vertices: np.ndarray, edges: np.ndarray, marked: set[int]) -> None:
        self.v, self.e, self.image_border_vertices = vertices, edges, np.asarray(sorted(marked), np.int64)


def test_the_mark_not_the_positions_decides_the_image_border():
    """A border junction moved half a pixel inside, as dw2d's relocation did before 0.8.0, is excluded by the mark."""
    mesh, _, _ = _double_bubble()
    vertices, edges = mesh
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    corners = np.array([[lo[0] - 5, lo[1] - 5], [lo[0] - 5, hi[1] + 5], [hi[0] + 5, lo[1] - 5], [hi[0] + 5, hi[1] + 5]])
    framed = np.vstack([vertices, corners])  # the four image corners, as dw2d keeps them
    junctions = sorted(j["vertex"] for j in fb2.junction_table((framed, edges), exclude_image_border=False)[0])
    assert len(junctions) == 2
    by_position = fb2.image_border_vertices((framed, edges))
    assert set(junctions).isdisjoint(by_position)  # control: by position, no junction is on the border
    marked = _MarkedMesh(framed, edges, set(by_position) | {junctions[0]})
    assert fb2.image_border_vertices(marked) == set(by_position) | {junctions[0]}
    _, report = fb2.infer_tensions(marked, diagnostics=True)
    assert report.image_border_from == "dw2d mark"
    assert report.n_junctions_excluded == 1
    _, report = fb2.infer_tensions((framed, edges), diagnostics=True)
    assert report.image_border_from == "vertex positions"
    assert report.n_junctions_excluded == 0
    _, report = fb2.infer_pressures(marked, dict.fromkeys([(0, 1), (0, 2), (1, 2)], 1.0), diagnostics=True)
    assert report.image_border_from == "dw2d mark"


def test_a_mesh_whose_mark_is_none_is_read_by_position():
    mesh, _, _ = _double_bubble()
    vertices, edges = mesh

    class Unmarked:
        v, e, image_border_vertices = vertices, edges, None

    with pytest.warns(UserWarning, match="image corners"):
        assert fb2.image_border_vertices(Unmarked()) == set()
