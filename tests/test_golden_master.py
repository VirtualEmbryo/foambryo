"""Golden-master regression test for foambryo's tension/pressure inference, on meshes committed with the tests.

Written at the package's initial validation stage. The golden master is
`Examples/reference_tensions.txt` / `reference_pressures.txt`, generated (`Examples/reference.ipynb`)
from `data/Meshes/Benchmark_original_mesh.rec` — external data not present locally (see
`tests/conftest.py`). We substitute `benchmarking-dataset/000_mesh.rec`, which
investigation showed reproduces the existing reference files to floating-point precision,
so this test targets the actual checked-in reference files, not new ones.

Tolerances: the original notebook-based check (an example notebook, removed in 0.5.0; this test
replaces it) used exact equality for tensions and `abs(diff) < 1e-3` for pressures. Re-running today (newer
numpy/scipy than when the reference was generated) no longer reproduces tensions to
exact bit equality — max observed relative difference is ~1.6e-14 for tensions and
~1.9e-7 for pressures, both far inside the original notebook's own pressure tolerance.
We use `rtol=1e-8` (tensions) and `1e-3` absolute (pressures, matching the original
notebook) so this test is robust to that library-version-induced floating-point noise
while still catching any real regression. Measured again on the 0.7.1 tree (October 2026, numpy 2.0 to 2.5,
scipy 1.13 to 1.18, identical on all four): 3.0e-9 for tensions and 6.0e-7 for pressures, larger than the
first measurement but inside both tolerances. They moved with the exact gauge of the tension
solves (0.7.0; 1.6e-14 on the commit before it, 3.0e-9 from it on): the reference files were
written under the earlier soft normalisation row: their mean is 1 - 2.97e-9, and the exact gauge
rescales every tension by that factor.

The meshes are read from `tests/data/golden_master_meshes/` (the benchmarking dataset's files, byte for
byte, compressed), so the golden master runs without the dataset, in the CI included. Two more cases pin
the same default inference on a two-cell (002) and a ten-cell (011) tissue, against values written by
`tests/data/make_golden_master_values.py` (relative tolerance 1e-8 on tensions and pressures; across
numpy 2.0 to 2.5 and scipy 1.13 to 1.18 they moved by at most 7.5e-15).
"""

import json

import pytest

from foambryo import dcel_mesh_from_file, infer_forces
from tests.conftest import (
    GOLDEN_MASTER_VALUES,
    REFERENCE_PRESSURES,
    REFERENCE_TENSIONS,
    golden_master_mesh,
    read_label_pair_value_file,
    read_label_value_file,
)

PINNED_CASES = ("002", "011")


def _infer(case: str, tmp_path_factory) -> tuple:
    path = golden_master_mesh(case, tmp_path_factory.mktemp(f"golden_master_{case}"))
    return infer_forces(dcel_mesh_from_file(str(path)))


@pytest.fixture(scope="module")
def inferred_forces(tmp_path_factory):
    return _infer("000", tmp_path_factory)


def test_tensions_match_reference(inferred_forces):
    tensions, _ = inferred_forces
    reference = read_label_pair_value_file(REFERENCE_TENSIONS)

    assert {(int(a), int(b)) for a, b in tensions} == set(reference)
    for key, expected in reference.items():
        got = tensions[key]
        assert got == pytest.approx(expected, rel=1e-8), f"interface {key}: {got} != {expected}"


def test_pressures_match_reference(inferred_forces):
    _, pressures = inferred_forces
    reference = read_label_value_file(REFERENCE_PRESSURES)

    assert {int(c) for c in pressures} == set(reference)
    for key, expected in reference.items():
        got = pressures[key]
        assert got == pytest.approx(expected, abs=1e-3), f"cell {key}: {got} != {expected}"


@pytest.mark.parametrize("case", PINNED_CASES)
def test_default_inference_matches_its_pinned_values(case, tmp_path_factory):
    """The default tensions and pressures of a two-cell and a ten-cell benchmark tissue, to a relative 1e-8."""
    pinned = json.loads(GOLDEN_MASTER_VALUES.read_text())["cases"][case]
    tensions, pressures = _infer(case, tmp_path_factory)

    expected_tensions = {(a, b): value for a, b, value in pinned["tensions"]}
    assert {(int(a), int(b)) for a, b in tensions} == set(expected_tensions)
    for key, expected in expected_tensions.items():
        assert tensions[key] == pytest.approx(expected, rel=1e-8), f"{case} interface {key}"
    expected_pressures = dict(pinned["pressures"])
    assert {int(c) for c in pressures} == set(expected_pressures)
    for key, expected in expected_pressures.items():
        assert pressures[key] == pytest.approx(expected, rel=1e-8, abs=1e-12), f"{case} cell {key}"
