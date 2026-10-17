"""Write `angle_reading_reference_values.json`, the numbers behind the bit-identity baseline.

`tests/test_second_order_angle_reading.py` has two gates on the default angle reading: a bit-identity
gate against `angle_reading_bit_identity_baseline.json` (SHA-256 digests, so only meaningful on the
numerical stack it was verified on), and a 1e-12 tolerance gate that runs on every stack. This script
writes the tolerance gate's reference values. It computes, in one run, both the digests and the values
of every quantity on the 47 `benchmarking-dataset` reference meshes, and **refuses to write unless every
digest equals the baseline's**, so the values written are the baseline's own numbers and not merely
those of whatever tree it runs on. Run it on the stack the test names (`BITWISE_NUMPY`,
`BITWISE_SCIPY`), from the root of a foambryo checkout, with the sibling `benchmarking-dataset/`:

    PYTHONPATH=src:. python tests/data/make_angle_reading_reference_values.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy

from tests import test_second_order_angle_reading as gate


def main() -> int:
    """Measure, check every digest against the baseline, and write the values; exit 1 on any mismatch."""
    if (np.__version__, scipy.__version__) != (gate.BITWISE_NUMPY, gate.BITWISE_SCIPY):
        print(
            f"STOP: run on numpy {gate.BITWISE_NUMPY} / scipy {gate.BITWISE_SCIPY}, not "
            f"{np.__version__} / {scipy.__version__}",
            file=sys.stderr,
        )
        return 1
    baseline = json.loads(gate.BASELINE.read_text())["cases"]
    cases, mismatches = {}, []
    for path in sorted(gate.DATASET_DIR.glob("*_mesh.rec")):
        case = path.stem.removesuffix("_mesh")
        digests, values = gate._measure_one_case(str(path))  # the gate's own measurement
        mismatches += [f"{case}/{field}" for field, value in digests.items() if baseline[case][field] != value]
        cases[case] = values
    if mismatches or set(cases) != set(baseline):
        print(f"STOP: {len(mismatches)} digest(s) differ from the baseline: {mismatches[:12]}", file=sys.stderr)
        return 1
    out = Path(gate.REFERENCE_VALUES)
    out.write_text(
        json.dumps(
            {
                "what": "The default angle reading's numbers on the 47 benchmarking-dataset reference meshes: every "
                "method's tensions per interface, and count, sum and sum of squares of the angles (radians), "
                "lengths and per-edge angles. Written by make_angle_reading_reference_values.py in a run whose "
                "digests all equal angle_reading_bit_identity_baseline.json.",
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "n_cases": len(cases),
                "cases": cases,
            },
            indent=0,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"wrote {out.name}: {len(cases)} cases, every digest equal to the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
