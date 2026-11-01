"""Write `golden_master_values.json`: the default inference's tensions and pressures on two committed meshes.

`tests/test_golden_master.py` checks the default `infer_forces` (Young-Dupré tensions, variational
pressures) on three benchmark meshes committed with the tests (`tests/data/golden_master_meshes/`).
Case 000 is checked against the published reference files (`Examples/reference_tensions.txt`,
`reference_pressures.txt`); cases 002 and 011 against the values this script writes, computed by the
code of the checkout it runs from. Run it from the root of a foambryo checkout whose `src/` is the
code to pin, only when a change of these numbers is deliberate and reviewed:

    PYTHONPATH=src:. python tests/data/make_golden_master_values.py
"""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import foambryo
from foambryo import dcel_mesh_from_file, infer_forces
from tests.conftest import GOLDEN_MASTER_MESHES, GOLDEN_MASTER_VALUES, golden_master_mesh

CASES = ("002", "011")


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    # A fixed git command on this checkout.
    command = ["git", "-C", str(root), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def main() -> int:
    """Infer on each case and write the values with their provenance; refuse if foambryo is not this checkout's."""
    root = Path(__file__).resolve().parents[2]
    if Path(foambryo.__file__).resolve().parents[1] != root / "src":
        print(f"STOP: foambryo imported from {foambryo.__file__}, not from {root / 'src'}", file=sys.stderr)
        return 1
    cases = {}
    with tempfile.TemporaryDirectory() as directory:
        for case in CASES:
            path = golden_master_mesh(case, Path(directory))
            tensions, pressures = infer_forces(dcel_mesh_from_file(str(path)))
            cases[case] = {
                "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensions": [[int(a), int(b), float(v)] for (a, b), v in sorted(tensions.items())],
                "pressures": [[int(c), float(v)] for c, v in sorted(pressures.items())],
            }
    record = {
        "what": "foambryo.infer_forces with its defaults (Young-Dupre tensions with mean 1, variational pressures "
        "with base pressure 0) on benchmark meshes 002 and 011, read from tests/data/golden_master_meshes/. "
        "Tensions are relative (mean 1); pressures are in tension units per voxel.",
        "written_by": "tests/data/make_golden_master_values.py",
        "foambryo_commit": _git("rev-parse", "HEAD"),
        "foambryo_src_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", "src")),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {p: metadata.version(p) for p in ("numpy", "scipy", "networkx", "trimesh")},
        "meshes": str(GOLDEN_MASTER_MESHES.relative_to(root)),
        "cases": cases,
    }
    GOLDEN_MASTER_VALUES.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {GOLDEN_MASTER_VALUES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
