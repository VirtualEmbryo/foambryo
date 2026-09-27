"""Golden-master regression test for foambryo's tension/pressure inference.

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
while still catching any real regression.
"""

import pytest

from foambryo import dcel_mesh_from_file, infer_forces
from tests.conftest import (
    REFERENCE_PRESSURES,
    REFERENCE_TENSIONS,
    read_label_pair_value_file,
    read_label_value_file,
    require_substitute_mesh,
)


@pytest.fixture(scope="module")
def inferred_forces():
    mesh_path = require_substitute_mesh()
    mesh = dcel_mesh_from_file(str(mesh_path))
    return infer_forces(mesh)


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
