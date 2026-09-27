#!/usr/bin/env python
r"""CLI: why the Young-Dupre family is 1000x more accurate than the force-balance family here.

The measurement that closes the question of the Variational/Young-Dupre inversion. The
dataset audit found, gauge-matched, that on the 42 equilibrium
`benchmarking-dataset` meshes `YoungDupre` reaches **2.3e-05** while `Variational` reaches
**2.1e-02**, and that the ordering *inverts* on the analytic doublet. Two hypotheses were
on the table:

- **H2 (discretisation)** — the doublet's single triple line is under-resolved, penalising
  the angle-based family. Tested by `foambryo_benchmarks/run_doublet_convergence.py`.
- **H1 (generator mismatch)** — the simulation meshes are at the discrete equilibrium of
  whatever energy their generator minimised, which is not the operator foambryo inverts.

This script measures the three things needed to decide between them, none of which requires
knowing what the generator was:

1. **Dimensionless resolution.** Edges per triple line (`junction length / mean edge
   length`) and `sqrt(interface area) / mean edge length`, for the dataset meshes and for the
   doublet at each `n_rings`. H2 requires the dataset meshes to be *better* resolved than the
   doublet; this says whether they are.
2. **Angle consistency.** How far each mesh's measured wedge angles are from the Neumann
   angles implied by its *nominal* tensions, in degrees. If a generator had pinned the
   junction angles to machine precision, this would be ~1e-3 degrees.
3. **Angle-noise calibration — the decisive one.** Feed the *exact* Neumann angles implied by
   the nominal tensions into the same matrix builder `YoungDupre` uses, then perturb those
   angles by *independent* Gaussian noise of the same magnitude as the deviation measured in
   (2), and see what tension error results. Comparing that against the *actual* error
   separates two possibilities that (2) alone cannot: angle deviations that are effectively
   random, and angle deviations that are structured so as to leave the inferred tensions
   almost untouched — i.e. nearly orthogonal to the column space of the Young-Dupre operator.

**Two controls that make (3) readable rather than merely suggestive.**

- `young_dupre_error_exact_neumann_angles` is the error with *no* angle perturbation at all,
  which bounds everything else. It is not always zero, and its value is informative: it comes
  out at exactly `|1 - mean(gamma_nominal)|` — 5.56e-06 on case `000`, 5.88e-06 on `011`, and
  ~1e-15 on `008`/`031`/`044` — because the builder's normalisation row asks for `mean = 1`
  while the dataset's tensions are stored rounded to 4 decimals and so have a mean of
  0.9999944 / 1.0000059. So this floor is the ground-truth *file's* gauge offset, not an
  inconsistency in the mesh, and `YoungDupre`'s measured 2.3e-05 sits only ~4x above it.
- `wedge_sum_max_error_deg`: the three measured wedges of each junction must sum to 360
  degrees. Measured max error over these meshes is ~1e-13 degrees, so `compute_angles_tri`
  returns a self-consistent wedge triple here and the supplement ambiguity that the
  angle-convention audit (`foambryo_benchmarks/run_angle_convention.py`) investigates is *not*
  firing on this dataset. Reported so that this measurement does not silently depend on that
  question's outcome.

**Caveat, stated because it is what the numbers rest on.** The injected noise is i.i.d. per
junction, while real discretisation error is spatially correlated. So the ratio
`actual error / noise-equivalent error` is a *bound* on how benign the real deviations are
relative to unstructured noise of the same size, not a decomposition of them.

Usage:
    uv run python -m foambryo_benchmarks.run_angle_consistency
    uv run python -m foambryo_benchmarks.run_angle_consistency --case-ids 000 044 --n-draws 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import metrics as m
from foambryo_benchmarks import phantoms as ph
from foambryo_benchmarks.run_case import _json_safe
from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions
from foambryo.dcel import DcelData
from foambryo.tension_inference import _build_matrix_tension_symmetrical_yd
from foambryo.validation import mesh_io

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
BASELINE_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline"

DEFAULT_DOUBLET_N_RINGS = (8, 16, 24, 32, 64)
DEFAULT_N_DRAWS = 20


def neumann_wedges(gamma_ab: float, gamma_bc: float, gamma_ac: float) -> tuple[float, float, float]:
    """The three wedge angles (radians) at a triple line, from its three tensions.

    Force balance at the line is `gamma_ab t_ab + gamma_bc t_bc + gamma_ac t_ac = 0` with unit
    in-plane directions `t`. The wedge in region `a` is the angle between `t_ab` and `t_ac`, so
    `|gamma_ab t_ab + gamma_ac t_ac| = gamma_bc` gives, by the law of cosines,

        cos(wedge_a) = (gamma_bc^2 - gamma_ab^2 - gamma_ac^2) / (2 gamma_ab gamma_ac),

    and cyclically. The three sum to `2 pi` whenever the tensions satisfy the triangle
    inequality, which is exactly the condition for the junction to be stable.

    Args:
        gamma_ab (float): Tension of the interface between regions a and b.
        gamma_bc (float): Tension of the interface between regions b and c.
        gamma_ac (float): Tension of the interface between regions a and c.

    Returns:
        tuple[float, float, float]: `(wedge_a, wedge_b, wedge_c)` in radians.
    """

    def wedge(opposite: float, side_1: float, side_2: float) -> float:
        cosine = (opposite**2 - side_1**2 - side_2**2) / (2 * side_1 * side_2)
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return (
        wedge(gamma_bc, gamma_ab, gamma_ac),
        wedge(gamma_ac, gamma_ab, gamma_bc),
        wedge(gamma_ab, gamma_ac, gamma_bc),
    )


def _ideal_angle_dict(
    triples: list[tuple[int, int, int]],
    truth: dict[tuple[int, int], float],
) -> dict[tuple[int, int, int], float]:
    """Build the `dict_rad` the mesh *would* have if it satisfied Neumann at nominal tensions.

    Only the three key permutations `_build_matrix_tension_symmetrical_yd` actually reads are
    emitted, with `a < b < c`: `(a, b, c)` is the wedge in `b`, `(b, a, c)` the wedge in `a`,
    `(a, c, b)` the wedge in `c` — the convention that function's own comments state.
    """
    ideal: dict[tuple[int, int, int], float] = {}
    for a, b, c in triples:
        wedge_a, wedge_b, wedge_c = neumann_wedges(truth[a, b], truth[b, c], truth[a, c])
        ideal[b, a, c] = wedge_a
        ideal[a, b, c] = wedge_b
        ideal[a, c, b] = wedge_c
    return ideal


def _solve_young_dupre(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length: dict[tuple[int, int, int], float],
    keys: list[tuple[int, int]],
) -> NDArray[np.float64]:
    """Run `YoungDupre`'s own builder and solver on a supplied angle dict."""
    matrix, vector = _build_matrix_tension_symmetrical_yd(dict_rad, dict_areas, dict_length, 1.0)
    tensions, *_ = linalg.lstsq(matrix, vector)
    del keys  # the solution is ordered by sorted(dict_areas), which the caller supplies
    return tensions


def resolution_metrics(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict:
    """Dimensionless resolution: edges per triple line and per interface diameter.

    Args:
        v (NDArray[np.float64]): Mesh vertices.
        f (NDArray[np.int64]): Mesh triangles.
        regions (NDArray[np.int64]): Per-triangle region pair.

    Returns:
        dict: `{"mean_edge_length", "edges_per_triple_line_median", "..._min",
            "interface_diameters_in_edges_median"}`.
    """
    _junctions, edge_to_triple = mesh_io.list_junctions(f, regions)
    edge_length = mesh_io.mean_edge_length(v, f)
    lengths = np.array(list(m.triple_line_lengths(v, edge_to_triple).values())) / edge_length
    areas = np.sqrt(np.array(list(m.interface_areas(v, f, regions).values()))) / edge_length
    return {
        "mean_edge_length": edge_length,
        "edges_per_triple_line_median": float(np.median(lengths)),
        "edges_per_triple_line_min": float(lengths.min()),
        "interface_diameters_in_edges_median": float(np.median(areas)),
    }


def angle_consistency(mesh: DcelData, truth: dict[tuple[int, int], float], n_draws: int) -> dict:
    """Measure angle deviation from Neumann, and calibrate it against i.i.d. angle noise.

    Args:
        mesh (DcelData): Mesh to analyse.
        truth (dict[tuple[int, int], float]): Nominal tension per interface.
        n_draws (int): Number of noise draws per noise level.

    Returns:
        dict: See the module docstring; `error_over_noise_equivalent` is the headline ratio.
    """
    rng = np.random.default_rng(0)
    dict_rad, _dict_deg, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    keys = sorted(dict_areas)
    gamma_truth = np.array([truth[key] for key in keys])

    triples = sorted({tuple(sorted(key)) for key in dict_rad})
    triples = [t for t in triples if all((t[i], t[j]) in truth for i, j in ((0, 1), (1, 2), (0, 2)))]
    ideal = _ideal_angle_dict(triples, truth)
    if set(ideal) != set(dict_rad):
        # Not an assertion: a mesh with a triple line whose interfaces are missing from the
        # ground truth is a legitimate input, and saying so beats silently comparing a
        # different junction set.
        return {"note": f"junction key sets differ ({len(ideal)} ideal vs {len(dict_rad)} measured)"}

    deviation = np.array([dict_rad[key] - ideal[key] for key in ideal])
    wedge_sums = [
        dict_rad[a, b, c] + dict_rad[b, a, c] + dict_rad[a, c, b] for a, b, c in triples
    ]

    def error_of(angles: dict) -> float:
        tensions = _solve_young_dupre(angles, dict_areas, dict_length, keys)
        return float(np.median(np.abs(tensions - gamma_truth) / np.abs(gamma_truth)))

    measured_error = error_of(dict_rad)
    exact_angle_error = error_of(ideal)

    noise_scale = float(np.median(np.abs(deviation)))
    noise_errors = []
    for _ in range(n_draws):
        noisy = dict(ideal)
        for a, b, c in triples:
            shift = rng.normal(0.0, noise_scale, 3)
            shift -= shift.mean()  # the three wedges of a junction must still sum to 2 pi
            noisy[b, a, c] = ideal[b, a, c] + shift[0]
            noisy[a, b, c] = ideal[a, b, c] + shift[1]
            noisy[a, c, b] = ideal[a, c, b] + shift[2]
        noise_errors.append(error_of(noisy))
    noise_equivalent = float(np.median(noise_errors))

    return {
        "n_triple_lines": len(triples),
        "angle_deviation_median_deg": float(np.degrees(noise_scale)),
        "angle_deviation_max_deg": float(np.degrees(np.abs(deviation).max())),
        "wedge_sum_max_error_deg": float(np.degrees(np.abs(np.array(wedge_sums) - 2 * np.pi)).max()),
        "young_dupre_error_measured_angles": measured_error,
        "young_dupre_error_exact_neumann_angles": exact_angle_error,
        "young_dupre_error_iid_noise_of_same_size": noise_equivalent,
        "error_over_noise_equivalent": measured_error / noise_equivalent if noise_equivalent else None,
        "n_draws": n_draws,
    }


def run(case_ids: list[str], n_draws: int, doublet_n_rings: tuple[int, ...]) -> dict:
    """Run both measurements on the named dataset cases and on the doublet at each resolution."""
    dataset: dict[str, dict] = {}
    for case_id in case_ids:
        mesh_path = DATASET_DIR / f"{case_id}_mesh.rec"
        if not mesh_path.exists():
            print(f"  SKIP {case_id}: {mesh_path} missing")
            continue
        truth = np.load(DATASET_DIR / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
        v, f, regions = mesh_io.read_rec(mesh_path)
        mesh = dcel_mesh_from_file(str(mesh_path))
        dataset[case_id] = {
            "group": case_groups.group_of(case_id),
            "resolution": resolution_metrics(v, f, regions),
            "angles": angle_consistency(mesh, truth, n_draws),
            "young_dupre_error_via_public_api": float(
                np.median(
                    [
                        abs(value - truth[key]) / abs(truth[key])
                        for key, value in infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre).items()
                        if key in truth
                    ],
                ),
            ),
        }
        entry = dataset[case_id]
        print(
            f"  {case_id} [{entry['group']}]: edges/triple-line median "
            f"{entry['resolution']['edges_per_triple_line_median']:.1f} | "
            f"angle dev {entry['angles'].get('angle_deviation_median_deg', float('nan')):.3f} deg | "
            f"YD error {entry['angles'].get('young_dupre_error_measured_angles', float('nan')):.2e} vs "
            f"i.i.d.-noise-equivalent "
            f"{entry['angles'].get('young_dupre_error_iid_noise_of_same_size', float('nan')):.2e} "
            f"(ratio {entry['angles'].get('error_over_noise_equivalent', float('nan')):.3f})",
        )

    doublet: dict[str, dict] = {}
    for n_rings in doublet_n_rings:
        v, f, regions = ph.doublet_mesh(1.0, n_rings=n_rings, n_phi=2 * n_rings)
        truth = ph.doublet_ground_truth_tensions(1.0, mean_tension=1.0)
        entry = {
            "resolution": resolution_metrics(v, f, regions),
            "angles": angle_consistency(DcelData(v, f, regions), truth, n_draws),
        }
        doublet[str(n_rings)] = entry
        print(
            f"  doublet n_rings={n_rings:4d}: edges/triple-line "
            f"{entry['resolution']['edges_per_triple_line_median']:.1f} | "
            f"angle dev {entry['angles']['angle_deviation_median_deg']:.3f} deg | "
            f"YD error {entry['angles']['young_dupre_error_measured_angles']:.2e} vs "
            f"i.i.d.-noise-equivalent {entry['angles']['young_dupre_error_iid_noise_of_same_size']:.2e} "
            f"(ratio {entry['angles']['error_over_noise_equivalent']:.3f})",
        )

    return {"dataset": dataset, "doublet": doublet}


def main() -> None:
    """Run from the command line and write `baseline/angle_consistency.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", type=str, nargs="*", default=["000", "008", "011", "031", "044"])
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--doublet-n-rings", type=int, nargs="*", default=list(DEFAULT_DOUBLET_N_RINGS))
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR)
    args = parser.parse_args()

    result = run(args.case_ids, args.n_draws, tuple(args.doublet_n_rings))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "angle_consistency.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
    )
    print(f"\nWrote angle_consistency.json to {args.output_dir}")


if __name__ == "__main__":
    main()
