r"""Write `two_dimensional_arc_reading_reference.json`: the arc reading's tensions and pressures, 18 golden meshes.

`tests/test_two_dimensional_arc_reading.py` checks `foambryo._two_dimensional.infer_tensions` (default arc reading)
and `infer_pressures` against the values the research implementation that defined the reading returns on the
meshes of `two_dimensional_reference_meshes.npz` (dw2d 0.1.5's meshes of the C. elegans example, three tissue masks
and ten exact equilibrium doublets): `arc_readings_2d.infer_l1(vertices, edges, "free", image_border_vertices)`
of foambryo's research record R188 (`research/R188-circular-arc-readings-for-2d-inference/scripts/`). That code
is not part of the package, so its output is stored. Run with the record's `scripts/` and its
`scripts/copied_instruments/` on the path, and this checkout's `src/`:

    PYTHONPATH=<R188 scripts>:<R188 scripts>/copied_instruments:src \
        python tests/data/make_two_dimensional_arc_reading_reference.py
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    """Compute and write the reference."""
    import arc_readings_2d as research
    import scipy

    from foambryo._two_dimensional import image_border_vertices

    reference = json.loads((HERE / "two_dimensional_reference_tensions.json").read_text())
    cases = {}
    with np.load(HERE / "two_dimensional_reference_meshes.npz") as arrays:
        for name in sorted(reference["cases"]):
            vertices, edges = arrays[f"{name}__vertices"], arrays[f"{name}__edges"]
            out = research.infer_l1(vertices, edges, "free", image_border_vertices((vertices, edges)))
            cases[name] = {
                "tensions": [[a, b, _finite_or_none(v)] for (a, b), v in sorted(out["tensions"].items())],
                "pressures": [[int(m), _finite_or_none(v)] for m, v in sorted(out["pressures"].items())],
                "n_junctions_used": len(out["used"]),
                "n_interfaces_read_by_their_chord": int(sum(f.rule == "chord_fallback" for f in out["fits"])),
            }
    record = {
        "what": "R188's research arc reading (L1-free) on the 18 golden meshes: tensions (mean 1 per junction-graph "
        "component) and pressures (tension / px, medium 0); null = not determined",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": cases,
    }
    (HERE / "two_dimensional_arc_reading_reference.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"written {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
