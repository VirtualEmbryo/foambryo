"""Shared pytest fixtures for foambryo.

Golden-master reference mesh — recorded decision, do not silently revisit
---------------------------------------------------------------------------
`SUBSTITUTE_MESH` below (`benchmarking-dataset/000_mesh.rec`) is **the** golden-master
reference mesh for this repository, by an explicit and *permanent* decision
(H. Turlier, taken at the package's initial validation stage).

It is **not** the published reference mesh. The published one is
`data/Meshes/Benchmark_original_mesh.rec`, from which
`Examples/reference_tensions.txt` / `reference_pressures.txt` were originally generated
(`Examples/reference.ipynb`); it is external data (git-LFS/Dropbox, see
`data/alternative_link_to_data`) and is not available in this environment.

The substitution is sound rather than merely expedient: `000_mesh.rec` reproduces the
existing, already-checked-in reference files to floating-point precision — max relative
difference ~1.6e-14 across all 18 tensions and ~1.9e-7 across all 7 pressures — so it is,
for every practical purpose, the same underlying benchmark mesh. The two files have never
been diffed byte-for-byte, however, so "the same mesh" is an inference from agreeing
outputs, not a proven identity.

A future reader should therefore not mistake `000_mesh.rec` for the published reference
mesh, and should not treat this as a temporary stand-in awaiting the original: the
decision is permanent unless `Benchmark_original_mesh.rec` itself resurfaces, in which
case the one worthwhile follow-up is to re-run `Examples/reference.ipynb` on it once and
confirm the agreement above.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
SUBSTITUTE_MESH = WORKSPACE_ROOT / "benchmarking-dataset" / "000_mesh.rec"
REFERENCE_TENSIONS = REPO_ROOT / "Examples" / "reference_tensions.txt"
REFERENCE_PRESSURES = REPO_ROOT / "Examples" / "reference_pressures.txt"
ORIGINAL_BENCHMARK_MESH = REPO_ROOT / "data" / "Meshes" / "Benchmark_original_mesh.rec"
FORCE_BALANCE_REFERENCE_IMPLEMENTATION = REPO_ROOT / "tests" / "reference" / "force_balance_reference_implementation.py"
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"


def require_substitute_mesh() -> Path:
    if not SUBSTITUTE_MESH.exists():
        pytest.skip(f"{SUBSTITUTE_MESH} not found (sibling benchmarking-dataset/ missing)")
    return SUBSTITUTE_MESH


def load_reference_implementation() -> ModuleType:
    """Load the force-balance reference implementation as a module, unmodified.

    `tests/reference/force_balance_reference_implementation.py` is a standalone numpy/scipy
    implementation of the force-balance operator and its solvers; `foambryo.validation` implements
    the same functions with the same algorithms. Loading it directly (rather than re-typing its
    logic in the test) is what makes the comparison in `test_validation_force_balance.py` a
    genuine check against the reference's *own* outputs, not against a second hand-written copy.
    It is part of the test tree (and of the source distribution), so the comparison runs wherever
    the tests do.
    """
    if not FORCE_BALANCE_REFERENCE_IMPLEMENTATION.exists():
        pytest.skip(f"{FORCE_BALANCE_REFERENCE_IMPLEMENTATION} not found (the test tree is incomplete)")
    spec = importlib.util.spec_from_file_location(
        "force_balance_reference_implementation",
        FORCE_BALANCE_REFERENCE_IMPLEMENTATION,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_label_pair_value_file(path: Path) -> dict[tuple[int, int], float]:
    """Parse `label_a label_b value` lines, as used by reference_tensions.txt."""
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        a, b, value = line.split()
        result[(int(a), int(b))] = float(value)
    return result


def read_label_value_file(path: Path) -> dict[int, float]:
    """Parse `label value` lines, as used by reference_pressures.txt."""
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label, value = line.split()
        result[int(label)] = float(value)
    return result
