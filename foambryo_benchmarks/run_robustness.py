#!/usr/bin/env python
r"""CLI: the robustness work — segmentation-perturbation & resolution robustness.

For a representative subset of cases (documented below, not a silent cap), for each of
several `dw3d` reconstruction variants, for each of several mask-perturbation conditions
(baseline, +-1 voxel dilate/erode, boundary noise, x2/x4 downsampling), reconstructs P2 from
the (possibly perturbed) mask and measures the same attribution triple
(`E_total`/`E_inference`/`E_meshing`, gauge-matched, `rho_free` on both populations, per-case
per-method) plus the spatial junction-instability proxy on P2
(`foambryo_benchmarks.junction_instability`).

**P1 (the ground-truth mesh) is never perturbed** — only the mask that feeds `dw3d` is, so
`E_inference` is identical across conditions for a given (case, method) by construction and
every degradation shown is attributable to the mesh, matching the attribution benchmark's own
finding on the unperturbed masks that meshing error dominates inference error (`E_inference` 2e-5
to 2e-2 against `E_total` 0.05-0.80 on the 42 equilibrium cases) -- by three to four orders of
magnitude for the angle-reading methods, but only by about 1.2-2.6x for `ForceBalanceKKT`/`Variational`
on the dithered meshes (`v0_3`, dw3d's default).

Typical invocation:

    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.run_robustness \\
        --out foambryo_benchmarks/baseline/robustness.json
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
from foambryo_benchmarks import junction_instability as ji
from foambryo_benchmarks import mesh_populations as pop
from foambryo_benchmarks import segmentation_perturbation as sperturb
from foambryo_benchmarks.run_attribution import METHOD_REGISTRY, _drop_per_key_detail, _json_safe
from foambryo import dcel_mesh_from_file, dcel_mesh_from_multimaterial_mesh
from foambryo.validation import mesh_io

#: Representative subset of the 42 equilibrium cases, one per distinct cell count present in the
#: dataset (3 through 11 cells: `002`=3, `008`=4, `003`=5, `001`=6, `000`=7, `006`=8, `004`=9
#: (integrity-flagged, kept by the recorded equilibrium-split decision), `030`=10, `011`=11 —
#: computed from each case's `*_dict_tensions.npy` label alphabet). **Stated explicitly, not a
#: silent cap**: running the full 42 x 6 conditions x 3 variants would be a further ~6x over the
#: attribution grid's already 6 minutes; this subset instead spans the dataset's entire cell-count
#: range so a degradation trend is not an artifact of one mesh size. `--cases` overrides it for a full run.
C2_CASE_SUBSET: tuple[str, ...] = ("002", "008", "003", "001", "000", "006", "004", "030", "011")

#: dw3d variants spanning the attribution work's interface-area-bias story: the published
#: baseline, the pre-link-checked (biased) former default, and the current shipped default
#: (signed median interface-area error -0.60 %, +15.73 % and +1.07 % on the 47 cases).
#: `a1b` omitted to hold the grid to 3 variants; it sits between `v0_3` and `a3_a5` on every
#: attribution metric and adds a 4th reconstruction per (case, condition) for no new qualitative point.
C2_VARIANT_NAMES: tuple[str, ...] = ("v0_3", "a3_a5", "offset_excluded_linkcheck")

#: Methods spanning the attribution benchmark's sensitivity split:
#: `YoungDupre`/`YoungDupreLocal` never touch `M` and were far less sensitive to mesh
#: quality; `ForceBalanceKKT` inverts `M` and was 5.5x more sensitive across the same
#: variant switch. Testing "does the ranking survive degradation" needs at least one of each.
C2_METHOD_NAMES: tuple[str, ...] = ("YoungDupre", "YoungDupreLocal", "ForceBalanceKKT")

#: One boundary-noise level, not a sweep — the robustness benchmark compares degradation *kinds*
#: (dilate/erode/noise/downsample), not noise *levels* (that would be a separate study).
BOUNDARY_NOISE_FRACTION = 0.2
BOUNDARY_NOISE_SEED_BASE = 20260801

CONDITIONS: tuple[str, ...] = ("baseline", "dilate1", "erode1", "boundary_noise", "downsample2", "downsample4")


def perturb_mask(mask: np.ndarray, condition: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply one named perturbation condition to a mask.

    Args:
        mask (np.ndarray): Segmentation mask.
        condition (str): One of `CONDITIONS`.
        rng (np.random.Generator): Used only by `"boundary_noise"`.

    Returns:
        tuple[np.ndarray, dict]: `(perturbed_mask, detail)`; `detail` is condition-specific
            (e.g. `n_voxels_changed` for boundary noise), empty for the deterministic ones.

    Raises:
        ValueError: If `condition` is not one of `CONDITIONS`.
    """
    if condition == "baseline":
        return mask, {}
    if condition == "dilate1":
        return sperturb.dilate_labels(mask, voxels=1), {}
    if condition == "erode1":
        return sperturb.erode_labels(mask, voxels=1), {}
    if condition == "boundary_noise":
        noisy, n_changed = sperturb.add_boundary_noise(mask, BOUNDARY_NOISE_FRACTION, rng)
        return noisy, {"n_voxels_changed": n_changed, "fraction": BOUNDARY_NOISE_FRACTION}
    if condition == "downsample2":
        return sperturb.downsample_labels_majority(mask, factor=2), {}
    if condition == "downsample4":
        return sperturb.downsample_labels_majority(mask, factor=4), {}
    message = f"unknown perturbation condition {condition!r}; expected one of {CONDITIONS}"
    raise ValueError(message)


def run_case(
    case_id: str,
    dataset_dir: Path,
    dw3d_module: object,
    variant_names: list[str],
    method_names: list[str],
    conditions: list[str],
    min_distance: int,
) -> list[dict]:
    """Run the full robustness grid (conditions x variants x methods) for one case.

    Args:
        case_id (str): Dataset case id.
        dataset_dir (Path): Directory holding the case's files.
        dw3d_module (object): The imported `dw3d` module (see `dw3d_variants.import_dw3d`).
        variant_names (list[str]): `dw3d` variant short names to run.
        method_names (list[str]): Keys of `foambryo_benchmarks.run_attribution.METHOD_REGISTRY`.
        conditions (list[str]): Perturbation condition names (`CONDITIONS`).
        min_distance (int): Passed to every `dw3d` reconstruction, unchanged across
            conditions — including the downsampled ones, deliberately: a real lower-
            resolution acquisition would still be fed to the pipeline at its default
            settings, and this benchmark measures the pipeline's robustness as shipped, not a re-tuned
            variant of it.

    Returns:
        list[dict]: One record per `(condition, variant, method)`.
    """
    mesh_path = dataset_dir / f"{case_id}_mesh.rec"
    tensions_path = dataset_dir / f"{case_id}_dict_tensions.npy"
    mask_path = dataset_dir / f"{case_id}_labels_filled.tif"

    dict_truth_tensions = np.load(tensions_path, allow_pickle=True).item()
    v1, f1, regions1 = mesh_io.read_rec(mesh_path)
    mesh_p1 = dcel_mesh_from_file(mesh_path)
    rho_free_p1 = am.rho_free_of_mesh(v1, f1, regions1)
    dict_tensions_p1_by_method = {name: METHOD_REGISTRY[name](mesh_p1) for name in method_names}

    mask = skimage.io.imread(mask_path)
    original_labels = set(np.unique(mask).tolist()) - {0}
    rng = np.random.default_rng(BOUNDARY_NOISE_SEED_BASE + int(case_id))

    records = []
    for condition in conditions:
        perturbed_mask, perturb_detail = perturb_mask(mask, condition, rng)
        perturbed_labels = set(np.unique(perturbed_mask).tolist()) - {0}
        dropped_labels = sorted(original_labels - perturbed_labels)
        if dropped_labels:
            perturb_detail["dropped_labels"] = dropped_labels

        for variant_name in variant_names:
            algorithm = dv.resolve_variant(dw3d_module, variant_name, min_distance=min_distance)
            t0 = time.perf_counter()
            points, triangles, labels = dv.reconstruct(algorithm, perturbed_mask)
            reconstruction_wall_time_s = time.perf_counter() - t0
            rho_free_p2 = am.rho_free_of_mesh(points, triangles, labels)
            mesh_p2 = dcel_mesh_from_multimaterial_mesh(points, triangles, labels)

            for method_name in method_names:
                base_record = {
                    "case": case_id,
                    "equilibrium_group": case_groups.group_of(case_id),
                    "condition": condition,
                    "perturb_detail": perturb_detail,
                    "dw3d_variant": variant_name,
                    "method": method_name,
                }

                try:
                    dict_tensions_p2 = METHOD_REGISTRY[method_name](mesh_p2)
                except Exception as exc:  # a per-cell failure must not stop the grid
                    records.append(
                        pop.tag(
                            {**base_record, "error": f"{type(exc).__name__}: {exc}"},
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

                try:
                    report = ji.junction_instability_report(mesh_p2, dict_tensions_p2)
                    instability = {
                        "n_candidate_triples": report["n_candidate_triples"],
                        "n_non_triple": report["n_non_triple"],
                        "n_present": report["n_present"],
                        "n_absent": report["n_absent"],
                        "n_unstable": report["n_unstable"],
                        "fraction_unstable": report["fraction_unstable"],
                        "spatial": ji.spatial_summary(report, n_bins=3),
                    }
                except ValueError as exc:
                    instability = {"error": f"{type(exc).__name__}: {exc}"}

                records.append(
                    pop.tag(
                        {
                            **base_record,
                            "reconstruction_wall_time_s": reconstruction_wall_time_s,
                            "rho_free_p1": rho_free_p1["rho_free"],
                            "rho_free_p2": rho_free_p2["rho_free"],
                            "n_interfaces_p1": rho_free_p1["n_interfaces"],
                            "n_interfaces_p2": rho_free_p2["n_interfaces"],
                            "instability": instability,
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
    default_out = Path(__file__).resolve().parent / "baseline" / "robustness.json"
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--cases", nargs="*", default=list(C2_CASE_SUBSET), help="Case ids to run.")
    parser.add_argument("--variants", nargs="*", default=list(C2_VARIANT_NAMES), help="dw3d variant names.")
    parser.add_argument("--methods", nargs="*", default=list(C2_METHOD_NAMES), help="Tension methods to run.")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), help="Perturbation conditions.")
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
    unknown_conditions = [c for c in args.conditions if c not in CONDITIONS]
    if unknown_conditions:
        message = f"unknown condition(s) {unknown_conditions}; available: {CONDITIONS}"
        raise SystemExit(message)

    print(f"dw3d: {getattr(dw3d_module, '__file__', '?')}")
    print(
        f"{len(args.cases)} cases x {len(args.conditions)} conditions x "
        f"{len(args.variants)} variants x {len(args.methods)} methods",
    )

    all_records = []
    for i, case_id in enumerate(args.cases):
        t0 = time.perf_counter()
        records = run_case(
            case_id,
            args.dataset_dir,
            dw3d_module,
            args.variants,
            args.methods,
            args.conditions,
            args.min_distance,
        )
        all_records.extend(records)
        n_errors = sum(1 for r in records if "error" in r)
        elapsed = time.perf_counter() - t0
        print(f"[{i + 1}/{len(args.cases)}] case {case_id}: {len(records)} records, {n_errors} errors, {elapsed:.1f}s")

    expected = len(args.cases) * len(args.conditions) * len(args.variants) * len(args.methods)
    if len(all_records) != expected:
        message = f"expected {expected} records, got {len(all_records)} - the grid did not complete as specified"
        raise RuntimeError(message)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_records = [_drop_per_key_detail(r) for r in all_records]
    args.out.write_text(json.dumps(_json_safe(all_records), indent=2))
    print(f"wrote {len(all_records)} records to {args.out}")


if __name__ == "__main__":
    main()
