#!/usr/bin/env python
r"""CLI: the attribution harness (meshing error versus inference error).

For each case, each `dw3d` reconstruction variant and each tension-inference method, emits
one JSON record with `E_total`, `E_inference`, `E_meshing` (all gauge-matched, the rule every
accuracy comparison in this package follows) and `rho_free` on both the P1 (simulator ground truth) and P2 (`dw3d`
reconstruction) meshes:

    mask --dw3d--> P2 mesh --foambryo--> gamma_hat(P2)   -> E_total     = ||gamma_hat(P2) - gamma*||
           P1 (ground-truth .rec)         --foambryo--> gamma_hat(P1)   -> E_inference = ||gamma_hat(P1) - gamma*||
                                                          gamma*                E_meshing   = E_total - E_inference

**The population-split work's non-additivity caveat applies to every row**: P1 is itself under-converged
(`rho_free` 0.0007-0.036 on the 42), so `E_meshing`'s sign is not a clean attribution -
`rho_free` is carried on both populations precisely so a reader can judge how much to
discount it. See `foambryo_benchmarks/attribution_metrics.py`'s module docstring.

This script produces the numbers; it draws no end-to-end verdict from them.

**dw3d resolution.** `dw3d` must be importable from the sibling `delaunay-watershed-3d`
checkout, not from any stale copy installed in this venv - see
`foambryo_benchmarks.dw3d_variants.import_dw3d`'s docstring. Typical invocation:

    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.run_attribution \\
        --dataset-dir ../benchmarking-dataset --out foambryo_benchmarks/baseline/attribution.json

The grid includes two synthetic "methods", `VariationalRouteP` and `VariationalRouteQ`,
which are `Variational` run explicitly through each solve route
(`foambryo.tension_inference.VariationalRoute`) rather than through the shipped default -
this is the direct test of the uncertainty work's open question ("route Q beats route P on the 42 but
is worse on the 5 non-equilibrium ones - is that still true on `dw3d` output, the
population the 5 most resemble?"). `VariationalRouteQ` is numerically the same computation
as `Variational` in the main grid (both use the shipped default route, Operator) and is
kept alongside it only so the two routes sit next to each other in one table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import skimage.io

from foambryo_benchmarks import attribution_metrics as am
from foambryo_benchmarks import case_groups
from foambryo_benchmarks import dw3d_variants as dv
from foambryo_benchmarks import mesh_populations as pop
from foambryo import (
    TensionComputationMethod,
    dcel_mesh_from_file,
    dcel_mesh_from_multimaterial_mesh,
    infer_tensions,
)
from foambryo.tension_inference import VariationalRoute, _infer_tension_variational_yd
from foambryo.validation import mesh_io

#: The attribution cross grid, plus the two explicit-route Variational variants that test whether
#: route Q still beats route P on `dw3d` reconstructions. Each value is
#: `(mesh) -> dict[(label_a, label_b), tension]`, mean_tension fixed at 1 everywhere.
METHOD_REGISTRY = {
    "YoungDupre": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.YoungDupre),
    "ForceBalanceKKT": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.ForceBalance),
    "YoungDupreLocal": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.YoungDupreLocal),
    "Variational": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.Variational),
    "BayesianMAP": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.ForceBalanceMAP),
    "VariationalRouteP": lambda mesh: _infer_tension_variational_yd(mesh, 1.0, route=VariationalRoute.Gram)[1],
    "VariationalRouteQ": lambda mesh: _infer_tension_variational_yd(mesh, 1.0, route=VariationalRoute.Operator)[1],
}

DEFAULT_METHOD_NAMES = tuple(METHOD_REGISTRY)


def discover_case_ids_with_masks(dataset_dir: Path) -> list[str]:
    """List every case id with a ground-truth mesh, ground-truth tensions AND a mask.

    The mask (`*_labels_filled.tif`) is what feeds `dw3d` (the `P1 -> ... -> mask -> dw3d
    -> P2` chain, see `foambryo_benchmarks.mesh_populations`); a case missing it cannot produce a
    P2 mesh and is excluded rather than silently skipped later.

    Args:
        dataset_dir (Path): Directory holding `{case_id}_mesh.rec` etc.

    Returns:
        list[str]: Sorted case ids.
    """
    ids = []
    for mesh_path in sorted(dataset_dir.glob("*_mesh.rec")):
        case_id = mesh_path.name.removesuffix("_mesh.rec")
        needed = (
            dataset_dir / f"{case_id}_dict_tensions.npy",
            dataset_dir / f"{case_id}_labels_filled.tif",
        )
        if all(path.exists() for path in needed):
            ids.append(case_id)
    return ids


def _json_safe(obj: object) -> object:
    """Recursively make a record JSON-serialisable (tuple keys, numpy scalars)."""
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _drop_per_key_detail(record: dict) -> dict:
    """Strip the per-interface `per_key` breakdown from `e_total_detail`/`e_inference_detail`.

    `attribution_errors` keeps `per_key` for interactive/debugging use, but committing it
    for all 1316 (case, variant, method) cells would make the baseline JSON ~15x larger
    than every other file under `foambryo_benchmarks/baseline/` for information `n`/`mean`/`median`/
    `p95` already summarise. Applied only when writing the committed baseline, not when
    `attribution_errors` is used directly (e.g. in tests).

    Args:
        record (dict): One attribution record, as returned by `run_case`.

    Returns:
        dict: `record` with `per_key` removed from both gauges of both `*_detail` blocks,
            if present.
    """
    for detail_key in ("e_total_detail", "e_inference_detail"):
        detail = record.get(detail_key)
        if not isinstance(detail, dict):
            continue
        for gauge in ("as_published", "gauge_matched"):
            if gauge in detail and isinstance(detail[gauge], dict):
                detail[gauge] = {k: v for k, v in detail[gauge].items() if k != "per_key"}
    return record


def run_case(
    case_id: str,
    dataset_dir: Path,
    dw3d_module: object,
    variant_names: list[str],
    method_names: list[str],
    min_distance: int,
) -> list[dict]:
    """Run the full attribution grid (variants x methods) for one case.

    Args:
        case_id (str): Dataset case id.
        dataset_dir (Path): Directory holding the case's files.
        dw3d_module (object): The imported `dw3d` module (see `dw3d_variants.import_dw3d`).
        variant_names (list[str]): `dw3d` variant short names to run.
        method_names (list[str]): Keys of `METHOD_REGISTRY` to run.
        min_distance (int): Passed to every `dw3d` reconstruction.

    Returns:
        list[dict]: One record per `(variant, method)`, tagged
            `population=P2_dw3d_reconstruction` (each record already carries the P1 side
            as `e_inference`, so it is one "attribution triple" per row, not two separate
            population records - see `attribution_metrics.attribution_errors`).
    """
    mesh_path = dataset_dir / f"{case_id}_mesh.rec"
    tensions_path = dataset_dir / f"{case_id}_dict_tensions.npy"
    mask_path = dataset_dir / f"{case_id}_labels_filled.tif"

    dict_truth_tensions = np.load(tensions_path, allow_pickle=True).item()
    v1, f1, regions1 = mesh_io.read_rec(mesh_path)
    mesh_p1 = dcel_mesh_from_file(mesh_path)
    rho_free_p1 = am.rho_free_of_mesh(v1, f1, regions1)

    dict_tensions_p1_by_method = {}
    for method_name in method_names:
        dict_tensions_p1_by_method[method_name] = METHOD_REGISTRY[method_name](mesh_p1)

    mask = skimage.io.imread(mask_path)

    records = []
    for variant_name in variant_names:
        algorithm = dv.resolve_variant(dw3d_module, variant_name, min_distance=min_distance)
        t0 = time.perf_counter()
        points, triangles, labels = dv.reconstruct(algorithm, mask)
        reconstruction_wall_time_s = time.perf_counter() - t0
        rho_free_p2 = am.rho_free_of_mesh(points, triangles, labels)
        mesh_p2 = dcel_mesh_from_multimaterial_mesh(points, triangles, labels)

        for method_name in method_names:
            try:
                dict_tensions_p2 = METHOD_REGISTRY[method_name](mesh_p2)
            except Exception as exc:  # a per-(variant, method) failure must not stop the grid
                records.append(
                    pop.tag(
                        {
                            "case": case_id,
                            "equilibrium_group": case_groups.group_of(case_id),
                            "dw3d_variant": variant_name,
                            "method": method_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        pop.P2_DW3D_RECONSTRUCTION,
                    ),
                )
                continue

            try:
                attribution = am.attribution_errors(
                    dict_tensions_p1_by_method[method_name],
                    dict_tensions_p2,
                    dict_truth_tensions,
                )
            except ValueError as exc:
                attribution = {"error": f"{type(exc).__name__}: {exc}"}

            records.append(
                pop.tag(
                    {
                        "case": case_id,
                        "equilibrium_group": case_groups.group_of(case_id),
                        "dw3d_variant": variant_name,
                        "method": method_name,
                        "reconstruction_wall_time_s": reconstruction_wall_time_s,
                        "rho_free_p1": rho_free_p1["rho_free"],
                        "rho_free_p2": rho_free_p2["rho_free"],
                        "n_interfaces_p1": rho_free_p1["n_interfaces"],
                        "n_interfaces_p2": rho_free_p2["n_interfaces"],
                        **attribution,
                    },
                    pop.P2_DW3D_RECONSTRUCTION,
                ),
            )
    return records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dataset_dir = Path(__file__).resolve().parent.parent.parent / "benchmarking-dataset"
    default_out = Path(__file__).resolve().parent / "baseline" / "attribution.json"
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--cases", nargs="*", default=None, help="Case ids to run; default: every case with a mask.")
    parser.add_argument("--variants", nargs="*", default=list(dv.DEFAULT_VARIANT_NAMES), help="dw3d variant names.")
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHOD_NAMES), help="Tension methods to run.")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    dw3d_module = dv.import_dw3d()
    available = dv.discover_variants(dw3d_module)
    unknown_variants = [name for name in args.variants if name not in available]
    if unknown_variants:
        message = f"unknown dw3d variant(s) {unknown_variants}; available: {sorted(available)}"
        raise SystemExit(message)
    unknown_methods = [name for name in args.methods if name not in METHOD_REGISTRY]
    if unknown_methods:
        message = f"unknown method(s) {unknown_methods}; available: {sorted(METHOD_REGISTRY)}"
        raise SystemExit(message)

    case_ids = args.cases if args.cases is not None else discover_case_ids_with_masks(args.dataset_dir)
    print(f"dw3d: {getattr(dw3d_module, '__file__', '?')}")
    print(f"{len(case_ids)} cases x {len(args.variants)} variants x {len(args.methods)} methods")

    all_records = []
    for i, case_id in enumerate(case_ids):
        t0 = time.perf_counter()
        records = run_case(case_id, args.dataset_dir, dw3d_module, args.variants, args.methods, args.min_distance)
        all_records.extend(records)
        n_errors = sum(1 for r in records if "error" in r)
        elapsed = time.perf_counter() - t0
        print(f"[{i + 1}/{len(case_ids)}] case {case_id}: {len(records)} records, {n_errors} errors, {elapsed:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_records = [_drop_per_key_detail(r) for r in all_records]
    args.out.write_text(json.dumps(_json_safe(all_records), indent=2))
    print(f"wrote {len(all_records)} records to {args.out}")


if __name__ == "__main__":
    main()
