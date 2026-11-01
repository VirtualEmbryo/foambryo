"""The golden-master meshes committed with the tests are the benchmarking dataset's files, byte for byte.

`tests/data/golden_master_meshes/` holds three of the dataset's meshes, gzip-compressed, so that
`tests/test_golden_master.py` runs without the dataset (in the CI). Where the sibling
`benchmarking-dataset/` is present, this checks the copies have not drifted from the originals, and
that the pinned values were written from these very files.
"""

import gzip
import hashlib
import json

import pytest

from tests.conftest import DATASET_DIR, GOLDEN_MASTER_CASES, GOLDEN_MASTER_MESHES, GOLDEN_MASTER_VALUES


def test_the_pinned_values_name_the_committed_meshes():
    pinned = json.loads(GOLDEN_MASTER_VALUES.read_text())["cases"]
    for case, values in pinned.items():
        committed = gzip.decompress((GOLDEN_MASTER_MESHES / f"{case}_mesh.rec.gz").read_bytes())
        assert hashlib.sha256(committed).hexdigest() == values["mesh_sha256"], case


@pytest.mark.parametrize("case", GOLDEN_MASTER_CASES)
def test_the_committed_mesh_is_the_dataset_file(case):
    original = DATASET_DIR / f"{case}_mesh.rec"
    if not original.exists():
        pytest.skip(f"{original} not found (sibling benchmarking-dataset/ missing)")
    committed = gzip.decompress((GOLDEN_MASTER_MESHES / f"{case}_mesh.rec.gz").read_bytes())
    assert committed == original.read_bytes()
