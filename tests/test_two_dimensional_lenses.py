"""Lenses in the 2D tension system: interfaces whose rows come from the same junctions only.

Pinned: on a cropped synthetic equilibrium foam (dw2d 0.8.0 at `min_dist` 3, its other defaults), a two-sided cell
whose two sides (3 and 2 vertices) are read by their chords has equal columns at its only junction, so the gauged
system is singular; the guard refuses it: its two tensions are `nan`, an `IllConditionedLensWarning` names them, the
report lists the lens, and every other tension stays finite and of order one. The closed-form condition number
agrees with an SVD; a lens whose angles separate its tensions is kept; the analytic equilibria keep every tension.
"""

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import JunctionGraphGaugeWarning

DATA = Path(__file__).resolve().parent / "data"
LENS = ((9, 14), (12, 14))


@pytest.fixture(scope="module")
def cropped_foam():
    """The mesh arrays and dw2d's image-border mark of the crop, as `infer_tensions` reads a dw2d `DCEL_Data`."""
    with np.load(DATA / "two_dimensional_mesh_of_a_cropped_foam_with_a_lens.npz") as z:
        return SimpleNamespace(v=z["vertices"], e=z["edges"], image_border_vertices=z["image_border_vertices"].tolist())


def test_the_lens_has_equal_columns_at_its_only_junction(cropped_foam):
    """The defect itself: read by their chords, the lens's two columns at its only junction are equal.

    Both sides run from vertex 146 to vertex 164, which is no junction (four materials on the crop's border); the
    wedge inside cell 14 is 0.
    """
    junctions, _ = fb2.junction_table(cropped_foam)
    at_146 = [j for j in junctions if j["vertex"] == 146]
    assert len(at_146) == 1
    assert at_146[0]["materials"] == (9, 12, 14)
    assert at_146[0]["wedges_rad"][14] < 1e-6  # arccos of a dot product of 1 to rounding
    assert not any(j["vertex"] == 164 for j in junctions)
    arcs = {a.materials: a for a in fb2.interface_arcs(cropped_foam) if a.materials in LENS}
    assert {a.fit for a in arcs.values()} == {"chord"}
    columns = fb2._young_dupre_columns(at_146[0])
    assert columns[LENS[0]] == pytest.approx(columns[LENS[1]], rel=0, abs=1e-15)  # equal to the last bit or two
    assert fb2._condition_of_two_columns(list(columns[LENS[0]]), list(columns[LENS[1]])) > 1e7


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_the_singular_lens_is_refused_with_nan_and_a_warning(cropped_foam, solver):
    with pytest.warns(fb2.IllConditionedLensWarning, match=r"\(9, 14\) and \(12, 14\)"):
        tensions, report = fb2.infer_tensions(cropped_foam, solver=solver, diagnostics=True)
    assert all(np.isnan(tensions[pair]) for pair in LENS)
    refused = {record["interfaces"]: record for record in report.refused_lenses}
    assert refused[LENS]["junction_vertices"] == (146,)
    assert refused[LENS]["condition_number"] == np.inf
    assert report.n_junctions_refused == sum(len(r["junction_vertices"]) for r in report.refused_lenses)
    largest = report.components[0]["index"]
    kept = [t for pair, t in tensions.items() if report.component_of[pair] == largest]
    assert kept
    assert all(np.isfinite(kept))
    finite = np.abs([t for t in tensions.values() if np.isfinite(t)])
    assert finite.max() < 10 * finite.mean()


def test_without_the_guard_the_lens_returned_two_opposite_huge_tensions(cropped_foam, monkeypatch):
    """The control: with the refusal disabled, the same mesh gives the singular solve's two opposite tensions."""
    monkeypatch.setattr(fb2, "MAX_LENS_CONDITION", np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        warnings.simplefilter("error", fb2.IllConditionedLensWarning)
        tensions = fb2.infer_tensions(cropped_foam, solver="dense")
    first, second = (tensions[pair] for pair in LENS)
    assert abs(first) > 1e6
    assert abs(second) > 1e6
    assert first * second < 0


def test_the_closed_form_condition_number_agrees_with_an_svd():
    rng = np.random.default_rng(196)
    for rows in (3, 6, 9):
        for _ in range(50):
            block = rng.normal(size=(rows, 2))
            singular = np.linalg.svd(block, compute_uv=False)
            closed = fb2._condition_of_two_columns(block[:, 0].tolist(), block[:, 1].tolist())
            assert closed == pytest.approx(singular[0] / singular[1], rel=1e-9)


def _junction(vertex, wedges):
    return {
        "vertex": vertex,
        "materials": (1, 2, 3),
        "wedges_rad": dict(zip((1, 2, 3), np.radians(wedges), strict=True)),
    }


@pytest.mark.parametrize(
    ("wedges", "refused"),
    [
        ((120.0, 120.0, 120.0), False),
        ((150.0, 150.0, 60.0), False),
        ((179.0, 179.0, 2.0), True),
        ((180.0, 180.0, 0.0), True),
    ],
)
def test_a_lens_is_refused_only_when_its_angles_cannot_separate_its_tensions(wedges, refused):
    """Three interfaces between the same two junctions (a lens, cell 3, between cells 1 and 2).

    The wedge inside the lens sets the conditioning: 120 and 60 degrees are kept, 2 degrees (condition number about
    57) and 0 refused.
    """
    used = [_junction(10, wedges), _junction(20, wedges)]
    kept, records = fb2._refuse_ill_conditioned_lenses(used)
    assert bool(records) is refused
    assert kept == ([] if refused else used)
    if refused:
        assert records[0]["junction_vertices"] == (10, 20)
        assert records[0]["interfaces"] == ((1, 2), (1, 3), (2, 3))


def test_nothing_changes_where_nothing_is_refused():
    used = [_junction(10, (120.0, 120.0, 120.0)), _junction(20, (110.0, 125.0, 125.0))]
    kept, records = fb2._refuse_ill_conditioned_lenses(used)
    assert records == []
    assert kept == used
    assert kept[0] is used[0]


@pytest.mark.parametrize("reading", ["arc", "chord"])
def test_the_analytic_equilibria_keep_every_tension(reading):
    from .test_two_dimensional_arc_reading import _double_bubble, _doublet

    for case in (_doublet, _double_bubble):
        mesh, true_tensions, _ = case()
        with warnings.catch_warnings():
            warnings.simplefilter("error", fb2.IllConditionedLensWarning)
            tensions, report = fb2.infer_tensions(
                mesh,
                angle_reading=reading,
                exclude_image_border=False,
                diagnostics=True,
            )
        assert report.refused_lenses == ()
        assert report.n_junctions_refused == 0
        assert all(np.isfinite(tensions[pair]) for pair in true_tensions)
