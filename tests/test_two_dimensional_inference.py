"""The 2D inference (`foambryo._two_dimensional`): foambryo-2d's per-junction Young-Dupré on foambryo's shared solve.

Pinned: on the 18 golden meshes of foambryo-2d (dw2d 0.1.5's meshes of the C. elegans example, three tissue
masks and ten exact equilibrium doublets), the chord reading (foambryo-2d's default) and the fitted-tangent reading
return foambryo-2d's tensions on the largest junction-graph component, the only one foambryo-2d solves (stored in
`tests/data/two_dimensional_reference_tensions.json`, written by
`tests/data/make_two_dimensional_reference_tensions.py`), to 1e-12 of the mean tension, with the same counts;
one junction with known angles gives the Young-Dupré (sine-law) tensions exactly; a mesh without any junction
is refused with `NoJunctionError`, an `UndeterminedInferenceError`; every junction-graph component is gauged
on its own and reported with its gauge, as in 3D; only Young-Dupré is offered; the image border is found
from the mesh alone, and a mesh without its corners warns and excludes nothing.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import TensionComputationMethod, UndeterminedInferenceError, UndeterminedTensionError
from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import JunctionGraphGaugeWarning

DATA = Path(__file__).resolve().parent / "data"
REFERENCE = json.loads((DATA / "two_dimensional_reference_tensions.json").read_text())
CASES = sorted(REFERENCE["cases"])
#: The report counts foambryo-2d and foambryo define alike; foambryo-2d's `n_rows`, `rank_deficiency` and
#: `n_interfaces_undetermined` are those of its largest component, compared with that component's below.
SHARED_REPORT_KEYS = (
    "n_junctions_used",
    "n_junctions_excluded",
    "n_junctions_nonfinite",
    "n_components",
    "n_higher_order_vertices",
)


@pytest.fixture(scope="module")
def meshes():
    with np.load(DATA / "two_dimensional_reference_meshes.npz") as arrays:
        return {name: (arrays[f"{name}__vertices"], arrays[f"{name}__edges"]) for name in CASES}


def test_every_golden_mesh_is_tested():
    assert len(CASES) == 18


@pytest.mark.parametrize("arm", ["default", "fitted_tangent"])
@pytest.mark.parametrize("name", CASES)
def test_the_tensions_are_those_of_foambryo_2d(meshes, name, arm):
    reference = REFERENCE["cases"][name][arm]
    reading = "fitted_tangent" if arm == "fitted_tangent" else "chord"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        tensions, report = fb2.infer_tensions(meshes[name], angle_reading=reading, diagnostics=True)
    expected = {(a, b): (np.nan if value is None else value) for a, b, value in reference["tensions"]}
    assert set(tensions) == set(expected)
    largest = {key for key, component in report.component_of.items() if component == 0}
    for key, value in expected.items():
        if key in largest:
            assert abs(tensions[key] - value) <= 1e-12, (key, tensions[key], value)
        else:
            # foambryo-2d determines only the largest component; foambryo gauges every other one on its own.
            assert np.isnan(value), key
            assert np.isnan(tensions[key]) == (report.component_of[key] is None), key
    counts = reference["report"]
    assert {key: getattr(report, key) for key in SHARED_REPORT_KEYS} == {key: counts[key] for key in SHARED_REPORT_KEYS}
    if report.components:
        assert report.components[0]["n_rows"] == counts["n_rows"]
        assert report.components[0]["rank_deficiency"] == counts["rank_deficiency"]
    assert len(tensions) - len(largest) == counts["n_interfaces_undetermined"]


def _one_junction(angles_deg: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Three straight interfaces from the origin, with the given angles inside materials 1, 2, 3 (sum 360)."""
    t_1, t_2, _ = np.radians(angles_deg)
    directions = [0.0, t_2, t_2 + t_1]  # interface (2,3) at 0, then (1,2) after material 2, then (1,3)
    vertices = np.array([[0.0, 0.0]] + [[10 * np.cos(d), 10 * np.sin(d)] for d in directions])
    edges = np.array([[0, 1, 2, 3], [0, 2, 1, 2], [0, 3, 1, 3]])
    return vertices, edges


def test_one_junction_gives_the_young_dupre_tensions():
    """Each tension is proportional to the sine of the angle opposite it: gamma_23 ~ sin(theta_1), and so on."""
    angles = (100.0, 130.0, 130.0)
    tensions, report = fb2.infer_tensions(
        _one_junction(angles),
        angle_reading="chord",
        exclude_image_border=False,
        diagnostics=True,
    )
    t_1, t_2, t_3 = np.radians(angles)
    expected = {(2, 3): np.sin(t_1), (1, 3): np.sin(t_2), (1, 2): np.sin(t_3)}
    scale = 3 / sum(expected.values())
    for key, value in expected.items():
        assert tensions[key] == pytest.approx(value * scale, rel=1e-12)
    assert report.n_junctions_used == 1
    assert report.components[0]["rank_deficiency"] == 0


def test_the_mesh_may_be_given_as_arrays_or_as_an_object(meshes):
    vertices, edges = meshes["c_elegans__md3"]

    class Mesh:
        v, e = vertices, edges

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        as_object, as_arrays = fb2.infer_tensions(Mesh()), fb2.infer_tensions((vertices, edges))
    assert as_object.keys() == as_arrays.keys()
    assert all(as_object[k] == as_arrays[k] or (np.isnan(as_object[k]) and np.isnan(as_arrays[k])) for k in as_object)


def test_a_mesh_without_a_junction_is_refused():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vertices = np.vstack([square, square + 3])
    edges = np.array(
        [
            [0, 1, 0, 1],
            [1, 2, 0, 1],
            [2, 3, 0, 1],
            [3, 0, 0, 1],
            [4, 5, 0, 2],
            [5, 6, 0, 2],
            [6, 7, 0, 2],
            [7, 4, 0, 2],
        ],
    )
    with pytest.raises(fb2.NoJunctionError) as caught:
        fb2.infer_tensions((vertices, edges))
    assert caught.value.interfaces == ((0, 1), (0, 2))
    assert isinstance(caught.value, UndeterminedInferenceError)
    assert isinstance(caught.value, ValueError)
    assert issubclass(UndeterminedTensionError, UndeterminedInferenceError)
    assert "expansion_labels" in str(caught.value)


def _two_clusters() -> tuple[np.ndarray, np.ndarray]:
    """Two separate clusters: one junction of materials 1, 2, 3 (three interfaces); junctions 4-5-6 and 4-5-7 (five)."""
    first_vertices, first_edges = _one_junction((100.0, 130.0, 130.0))
    second_vertices, second_edges = _one_junction((100.0, 130.0, 130.0))
    third_vertices, third_edges = _one_junction((110.0, 125.0, 125.0))
    vertices = np.vstack([first_vertices, second_vertices + 100, third_vertices + 200])
    relabel = np.array([0, 4, 5, 7])
    third_edges = np.column_stack([third_edges[:, :2] + 8, relabel[third_edges[:, 2:]]])
    edges = np.vstack([first_edges, second_edges + np.array([4, 4, 3, 3]), third_edges])
    return vertices, edges


def test_every_junction_graph_component_is_gauged_on_its_own_and_reported():
    """As in 3D: each component meets its own mean exactly, and the report says which interface is in which."""
    with pytest.warns(JunctionGraphGaugeWarning, match="2 component"):
        tensions, report = fb2.infer_tensions(
            _two_clusters(),
            angle_reading="chord",
            exclude_image_border=False,
            diagnostics=True,
        )
    assert report.n_components == 2
    assert report.n_interfaces_undetermined == 0
    largest = ((4, 5), (4, 6), (4, 7), (5, 6), (5, 7))
    smallest = ((1, 2), (1, 3), (2, 3))
    assert [component["n_interfaces"] for component in report.components] == [5, 3]
    assert [component["n_junctions"] for component in report.components] == [2, 1]
    assert {key: report.component_of[key] for key in largest + smallest} == {
        **dict.fromkeys(largest, 0),
        **dict.fromkeys(smallest, 1),
    }
    for index, members in enumerate((largest, smallest)):
        assert np.mean([tensions[key] for key in members]) == pytest.approx(1.0, rel=1e-12)
        assert "mean(gamma) = 1.0 exactly" in report.components[index]["gauge"]
        assert report.components[index]["rank_deficiency"] == 0
    # The single junction alone: its own gauge, the sine law.
    t_1, t_2, t_3 = np.radians((100.0, 130.0, 130.0))
    expected = {(2, 3): np.sin(t_1), (1, 3): np.sin(t_2), (1, 2): np.sin(t_3)}
    scale = 3 / sum(expected.values())
    for key, value in expected.items():
        assert tensions[key] == pytest.approx(value * scale, rel=1e-12)


def test_an_interface_no_used_junction_touches_is_undetermined():
    vertices, edges = _two_clusters()
    with pytest.warns(JunctionGraphGaugeWarning, match="3 undetermined"):
        tensions, report = fb2.infer_tensions(
            (vertices, edges),
            angle_reading="chord",
            exclude_image_border=False,
            excluded_vertices={0},
            diagnostics=True,
        )
    assert report.n_components == 1
    assert all(np.isnan(tensions[key]) for key in ((1, 2), (1, 3), (2, 3)))
    assert all(report.component_of[key] is None for key in ((1, 2), (1, 3), (2, 3)))
    assert report.n_interfaces_undetermined == 3


def test_only_young_dupre_is_offered():
    with pytest.raises(ValueError, match="YoungDupre"):
        fb2.infer_tensions(_one_junction((120.0, 120.0, 120.0)), mode=TensionComputationMethod.Lami)
    tensions = fb2.infer_tensions(
        _one_junction((120.0, 120.0, 120.0)),
        mode=TensionComputationMethod.YoungDupre,
        angle_reading="chord",
        exclude_image_border=False,
    )
    assert tensions == pytest.approx(dict.fromkeys(((1, 2), (1, 3), (2, 3)), 1.0), rel=1e-12)


def test_the_image_border_is_found_from_the_mesh_alone(meshes):
    vertices, _ = meshes["c_elegans__md3"]
    border = fb2.image_border_vertices(meshes["c_elegans__md3"])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    assert border == {i for i, (x, y) in enumerate(vertices) if x in (lo[0], hi[0]) or y in (lo[1], hi[1])}
    with pytest.warns(UserWarning, match="image corners"):
        assert fb2.image_border_vertices(_one_junction((120.0, 120.0, 120.0))) == set()


def test_an_unknown_reading_is_refused(meshes):
    with pytest.raises(ValueError, match="angle_reading"):
        fb2.infer_tensions(meshes["doublet_00__md3"], angle_reading="quadric")
    with pytest.raises(ValueError, match="angle_reading"):
        fb2.junction_table(meshes["doublet_00__md3"], angle_reading="quadric")
