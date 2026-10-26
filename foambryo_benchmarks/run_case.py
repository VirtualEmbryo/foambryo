#!/usr/bin/env python
r"""CLI: score one ground-truth mesh against its ground-truth tensions and pressures.

Given a `.rec` mesh + `dict_tensions.npy` + `dict_pressures.npy`, emit one JSON record
with tension/pressure error, `rho`, the singular spectrum and `sigma_soft` of the
force-balance operator, and wall time.

One row per (case x method): every existing `TensionComputationMethod` is run (this
harness adds no inference method of its own). Mesh
diagnostics (`rho`, spectrum, `sigma_soft`) are intrinsic to the mesh and ground-truth
tensions, independent of inference method, so they are computed once per case.

Gauge (the rule every accuracy comparison in this package follows): every tension and pressure
error is reported **twice** — as published, and after gauge matching (the inferred values
rescaled by one factor so their mean tension matches the ground truth's). See
`foambryo_benchmarks.metrics.gauge_matching_scale` for why the two differ even though both sides
nominally use `mean(gamma) = 1`.

`foambryo_benchmarks/run_baseline.py` runs this on the full 47-case `benchmarking-dataset` by
default; the early 3-case smoke-test restriction was lifted once the dataset had been audited
for equilibrium. The audit, and the recorded decision to exclude the five non-equilibrium
cases, live in `foambryo_benchmarks/run_dataset_audit.py` and `foambryo_benchmarks/case_groups.py`.

Usage:
    uv run python -m foambryo_benchmarks.run_case --case-id 000 --dataset-dir ../benchmarking-dataset
    uv run python -m foambryo_benchmarks.run_case --mesh path/to/foo_mesh.rec \\
        --tensions path/to/foo_dict_tensions.npy --pressures path/to/foo_dict_pressures.npy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import metrics as m
from foambryo import (
    ForceBalanceNormalisation,
    PressureComputationMethod,
    TensionComputationMethod,
    dcel_mesh_from_file,
    infer_force_balance,
    infer_pressures,
    infer_tensions,
)
from foambryo.validation import mesh_io


def _json_safe(obj: object) -> object:
    """Recursively make a record JSON-serialisable.

    Tuple dict keys become comma-joined strings and numpy scalars become Python scalars.
    `json.dumps`'s `default` hook only converts unrecognised *values*, not dict *keys*,
    so this has to walk the structure itself.
    """
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_case(mesh_path: Path, tensions_path: Path, pressures_path: Path) -> dict:
    """Run every `TensionComputationMethod` on one ground-truth mesh and measure error.

    Args:
        mesh_path (Path): Path to the ground-truth `.rec` mesh.
        tensions_path (Path): Path to the `dict_tensions.npy` ground truth.
        pressures_path (Path): Path to the `dict_pressures.npy` ground truth.

    Returns:
        dict: `{"mesh_diagnostics": {...}, "methods": {method_name: {...}}}`.
    """
    dict_truth_tensions = np.load(tensions_path, allow_pickle=True).item()
    dict_truth_pressures = np.load(pressures_path, allow_pickle=True).item()

    v, f, regions = mesh_io.read_rec(mesh_path)
    diagnostics = m.mesh_diagnostics(v, f, regions, dict_truth_tensions)

    mesh = dcel_mesh_from_file(str(mesh_path))

    methods = {}
    for tension_mode in TensionComputationMethod:
        t0 = time.perf_counter()
        try:
            dict_tensions = infer_tensions(mesh, mean_tension=1, mode=tension_mode)
            tension_wall_time = time.perf_counter() - t0
        except Exception as exc:
            methods[tension_mode.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        # Two gauges (the rule every accuracy comparison in this package follows): the
        # as-published error, and the error after matching the overall tension scale. The
        # soft normalisation row means `infer_tensions` does not return mean(gamma) exactly
        # 1, so the two differ; reporting only one of them is not acceptable.
        scale = m.gauge_matching_scale(dict_tensions, dict_truth_tensions)
        tension_error = m.tension_relative_errors(dict_tensions, dict_truth_tensions)
        tension_error_matched = m.tension_relative_errors(
            {key: scale * value for key, value in dict_tensions.items()},
            dict_truth_tensions,
        )

        t0 = time.perf_counter()
        try:
            dict_pressures = infer_pressures(
                mesh,
                dict_tensions,
                mode=PressureComputationMethod.Variational,
                base_pressure=0,
            )
            pressure_wall_time = time.perf_counter() - t0
            pressure_error = m.pressure_relative_errors(dict_pressures, dict_truth_pressures)
            # Same factor as the tensions: Delta_P = 2 gamma / R is linear in tension, so
            # one factor keeps the (tension, pressure) pair a consistent equilibrium.
            pressure_error_matched = m.pressure_relative_errors(
                {key: scale * value for key, value in dict_pressures.items()},
                dict_truth_pressures,
            )
            pressure_failure = None
        except Exception as exc:
            pressure_wall_time = None
            pressure_error = None
            pressure_error_matched = None
            pressure_failure = f"{type(exc).__name__}: {exc}"

        methods[tension_mode.name] = {
            "gauge_matching_scale": scale,
            "mean_inferred_tension": float(np.mean(list(dict_tensions.values()))),
            "tension_relative_error": tension_error,
            "tension_relative_error_matched_gauge": tension_error_matched,
            "tension_wall_time_s": tension_wall_time,
            "pressure_relative_error": pressure_error,
            "pressure_relative_error_matched_gauge": pressure_error_matched,
            "pressure_wall_time_s": pressure_wall_time,
            "pressure_error_note": pressure_failure,
        }

    return {
        "mesh_diagnostics": diagnostics,
        "methods": methods,
        "force_balance_normalisations": force_balance_diagnostics(mesh, dict_truth_tensions),
    }


def force_balance_diagnostics(
    mesh: object,
    dict_truth_tensions: dict[tuple[int, int], float],
) -> dict:
    """The force-balance work's normalisation comparison: exactness, conditioning and error bars per variant.

    The three `ForceBalance*` members of `TensionComputationMethod` already appear in
    `methods` above with their tension and pressure errors. What that block cannot carry is
    the *reason* the three variants exist: whether the gauge is exact, how well-conditioned the system
    actually solved is, and the per-interface posterior standard deviation. Those are
    recorded here, once per case, for all three variants.

    The error bars come with a calibration check that is the honest way to read them:
    `error_over_std` is the median of `|gamma_inferred - gamma_truth| / std`, gauge-matched.
    A well-calibrated i.i.d. error model would put this near 1; a value far above 1 says the
    residual is *systematic* (model or data mismatch) rather than noise, and that the error
    bars are optimistic. Reported rather than asserted.

    **On a `dw3d`-reconstructed mesh, read this block with the measured floor**: `rho_free` is
    about 50x the reference meshes' own value (median 0.546 against 0.0118 on the 40-case
    equilibrium cohort, every case above the 0.2 alarm threshold), so the three variants are
    fitting a geometry that no tension assignment balances. See
    `foambryo.force_balance_inference`'s module docstring.

    Args:
        mesh (object): The `DcelData` mesh (typed loosely to keep this module import-light).
        dict_truth_tensions (dict[tuple[int, int], float]): Ground-truth tension per interface.

    Returns:
        dict: One entry per `ForceBalanceNormalisation`, plus `n_interfaces`/`n_cells`.
    """
    out: dict[str, dict] = {}
    for normalisation in ForceBalanceNormalisation:
        result = infer_force_balance(mesh, mean_tension=1, normalisation=normalisation)
        truth = np.array([dict_truth_tensions[key] for key in result.interfaces])
        scale = float(np.mean(truth) / np.mean(result.tensions))
        absolute_error = np.abs(scale * result.tensions - truth)
        with np.errstate(divide="ignore", invalid="ignore"):
            calibration = absolute_error / (scale * result.tension_std)
        out[normalisation.name] = {
            "mean_inferred_tension": float(np.mean(result.tensions)),
            "mean_tension_deviation": result.mean_tension_deviation,
            "condition_number": result.condition_number,
            "residual_norm": result.residual_norm,
            "rho_at_inferred_tensions": result.rho,
            "sigma_f": result.sigma_f,
            "tension_std_median": float(np.median(result.tension_std)),
            "tension_std_max": float(np.max(result.tension_std)),
            "tension_std_per_key": result.dict_tension_std,
            "error_over_std_median": float(np.median(calibration[np.isfinite(calibration)])),
        }
    out["n_interfaces"] = len(next(iter(out.values()))["tension_std_per_key"])
    return out


def main() -> None:
    """Run one case from the command line and write its JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=None, help="Path to a ground-truth .rec mesh")
    parser.add_argument("--tensions", type=Path, default=None, help="Path to dict_tensions.npy")
    parser.add_argument("--pressures", type=Path, default=None, help="Path to dict_pressures.npy")
    parser.add_argument("--case-id", type=str, default=None, help="benchmarking-dataset case id, e.g. 000")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "benchmarking-dataset",
        help="Directory holding {case_id}_mesh.rec etc. (used with --case-id)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    if args.case_id is not None:
        mesh_path = args.dataset_dir / f"{args.case_id}_mesh.rec"
        tensions_path = args.dataset_dir / f"{args.case_id}_dict_tensions.npy"
        pressures_path = args.dataset_dir / f"{args.case_id}_dict_pressures.npy"
    else:
        if args.mesh is None or args.tensions is None or args.pressures is None:
            parser.error("either --case-id or all of --mesh/--tensions/--pressures must be given")
        mesh_path, tensions_path, pressures_path = args.mesh, args.tensions, args.pressures

    record = run_case(mesh_path, tensions_path, pressures_path)
    record["case"] = args.case_id if args.case_id is not None else mesh_path.stem

    text = json.dumps(_json_safe(record), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
