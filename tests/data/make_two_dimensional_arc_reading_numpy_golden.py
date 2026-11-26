"""Write `two_dimensional_arc_reading_numpy_golden.json`: the shipped arc reading's tensions and pressures, 18 meshes.

`tests/test_two_dimensional_circle_fit.py` checks `foambryo._two_dimensional.infer_tensions` (default arc reading)
and `infer_pressures` against these values to 1e-12 on every supported NumPy and SciPy: the circle fit is written
in NumPy without BLAS or LAPACK calls, so the fitted circles are the same bit for bit on every version, and only the
least-squares solves (SciPy's LAPACK) move the last bits. The meshes are those of
`two_dimensional_reference_meshes.npz`. Run from the repository root with this checkout's `src` on the path:

    PYTHONPATH=src python tests/data/make_two_dimensional_arc_reading_numpy_golden.py
"""

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    """Compute and write the golden master."""
    import scipy

    from foambryo import _two_dimensional as fb2
    from foambryo.junction_graph import JunctionGraphGaugeWarning

    reference = json.loads((HERE / "two_dimensional_reference_tensions.json").read_text())
    cases = {}
    with np.load(HERE / "two_dimensional_reference_meshes.npz") as arrays:
        for name in sorted(reference["cases"]):
            mesh = (arrays[f"{name}__vertices"], arrays[f"{name}__edges"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
                tensions = fb2.infer_tensions(mesh)
                pressures = fb2.infer_pressures(mesh, tensions)
            cases[name] = {
                "tensions": [[int(a), int(b), _finite_or_none(v)] for (a, b), v in sorted(tensions.items())],
                "pressures": [[int(m), _finite_or_none(v)] for m, v in sorted(pressures.items())],
            }
    record = {
        "what": "foambryo._two_dimensional's arc reading (NumPy circle fit) on the 18 golden meshes: tensions (mean 1 "
        "per junction-graph component, plain map) and pressures (tension / px, medium 0); null = not determined",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": cases,
    }
    (HERE / "two_dimensional_arc_reading_numpy_golden.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"written {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
