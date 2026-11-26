r"""Write `two_dimensional_reference_tensions.{npz,json}`: foambryo-2d's tensions on its 18 golden meshes.

`tests/test_two_dimensional_inference.py` checks `foambryo._two_dimensional.infer_tensions` against the
tensions foambryo-2d's `infer_tension` (default mode: per-junction Young-Dupré, exact gauge, image-border
junctions excluded) and `infer_tensions_per_junction(reading="fitted_tangent")` return on the meshes of
its golden master (dw2d 0.1.5's outputs on the C. elegans example, three tissue masks and ten exact
equilibrium doublets). foambryo cannot import foambryo-2d, so the reference is stored: the meshes'
vertices and edges (`.npz`) and, per mesh, the tensions and report (`.json`). Run with foambryo-2d's
`src/` and dw2d's `src/` on the path:

    PYTHONPATH=<foambryo-2d>/src:<dw2d>/src python tests/data/make_two_dimensional_reference_tensions.py \\
        <golden .npz> <golden .json>

where the two arguments are foambryo-2d's `tests/golden/dw2d_0.1.5_on_scipy_1.13_outputs.{npz,json}`.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT_KEYS = (
    "n_junctions_used",
    "n_junctions_excluded",
    "n_junctions_nonfinite",
    "n_components",
    "n_rows",
    "rank_deficiency",
    "n_interfaces_undetermined",
    "n_higher_order_vertices",
)


def main() -> int:
    """Read the golden meshes, run foambryo-2d on each, write the meshes and the reference tensions."""
    import foambryo2d
    from dw2d.Dcel import DCEL_Data

    golden_npz, golden_json = Path(sys.argv[1]), Path(sys.argv[2])
    arrays = np.load(golden_npz)
    cases = sorted(json.loads(golden_json.read_text())["cases"])
    meshes, reference = {}, {}
    for name in cases:
        vertices, edges = arrays[f"{name}__verts"], arrays[f"{name}__edges"]
        meshes[f"{name}__vertices"], meshes[f"{name}__edges"] = vertices, edges
        _, default, default_report = foambryo2d.infer_tension(DCEL_Data(vertices, edges), mean_tension=1)
        mesh = DCEL_Data(vertices, edges)
        fitted, fitted_report = foambryo2d.infer_tensions_per_junction(
            mesh,
            excluded_vertices=foambryo2d.image_border_vertices(mesh),
            reading="fitted_tangent",
        )
        reference[name] = {
            arm: {
                "tensions": [[a, b, None if not np.isfinite(t) else t] for (a, b), t in sorted(values.items())],
                "report": {key: report[key] for key in REPORT_KEYS},
            }
            for arm, values, report in (("default", default, default_report), ("fitted_tangent", fitted, fitted_report))
        }
    np.savez_compressed(HERE / "two_dimensional_reference_meshes.npz", **meshes)
    root = Path(foambryo2d.__file__).resolve().parents[2]
    command = ["git", "-C", str(root), "rev-parse", "HEAD"]
    commit = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603
    record = {
        "what": "foambryo-2d's tensions (mean 1, nan as null) and reports on its 18 golden meshes: infer_tension's "
        "default, and infer_tensions_per_junction with reading='fitted_tangent' and the image border excluded.",
        "written_by": "tests/data/make_two_dimensional_reference_tensions.py",
        "foambryo_2d_commit": commit,
        "numpy": np.__version__,
        "cases": reference,
    }
    (HERE / "two_dimensional_reference_tensions.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
