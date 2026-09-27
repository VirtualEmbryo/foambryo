#!/usr/bin/env python
r"""CLI: fit the mesh-convergence order of every tension method on the analytic doublet.

The dataset audit measured, gauge-matched, an *inversion* between two families of methods:

| | analytic doublet (`n_rings = 24`) | 42 simulation meshes |
|---|---|---|
| `YoungDupre` | 2.3e-02 | **2.3e-05** |
| `Variational` | **2.8e-05** | 2.1e-02 |

Three orders of magnitude, in both directions. Two hypotheses were advanced. This script
tests the second one, which is the cheap and decisive half:

> **H2 (discretisation).** The doublet has a *single* circular triple line, so its junction
> angles are poorly resolved at low resolution, penalising the angle-based family. The
> simulation meshes have many well-resolved junctions. If H2 dominates, refining the
> doublet should drive `YoungDupre`'s error down towards `Variational`'s; if the error
> plateaus instead, H2 does not explain the inversion and H1 (the generator minimised a
> *different* discrete energy than the operator being inverted) is left carrying it.

What is measured and how the order is fitted
--------------------------------------------
For each `n_rings` in the sweep, the exact analytic doublet mesh is rebuilt and every
`TensionComputationMethod` is run on it. The error reported is the **gauge-matched** median
relative tension error over the three interfaces (the rule every accuracy comparison in this
package follows): the ground truth
is generated at `mean_tension = 1`, and each method's output is rescaled by
`mean(truth) / mean(inferred)` before the error is taken, so no part of the number is the
gauge convention.

The mesh spacing is `h ~ 1 / n_rings`, so an error `~ C h^q` is a straight line of slope
`-q` in `log(error)` vs `log(n_rings)`. `q` is fitted by ordinary least squares over the
sweep, with a standard error from the residuals and an `R^2` — reported so that a *bad*
fit (a plateau, or a non-monotone curve) is visible as such rather than being reduced to a
single misleading exponent. The fit uses the last `--fit-last` resolutions (default: all),
since the coarsest point is often outside the asymptotic regime.

Resolution has two independent knobs, and conflating them would answer the wrong question
-----------------------------------------------------------------------------------------
`doublet_mesh(n_rings, n_phi)` resolves the *domes* with `n_rings` latitude bands and the
*rim* — the single triple line — with `n_phi` azimuthal segments. The dihedral angle at the
rim is limited by the polar extent of the band adjacent to it, i.e. by `n_rings`; the
triple line's own arc-length sampling is set by `n_phi`. Two modes are therefore run:

- `uniform` (default, the headline): `n_phi = 2 * n_rings`, so `h` shrinks isotropically.
  This is the honest convergence study, and it matches the shipped default
  `(n_rings, n_phi) = (24, 48)`.
- `fixed_phi`: `n_phi` held at `--fixed-phi` (default 48) while `n_rings` grows. Comparing
  the two isolates *which* of the two spacings each method's error actually tracks — a free
  extra discriminator, since a method limited by junction sampling will stall in
  `fixed_phi` while one limited by surface sampling will not.

Usage:
    uv run python -m foambryo_benchmarks.run_doublet_convergence
    uv run python -m foambryo_benchmarks.run_doublet_convergence --n-rings 8 16 32 --ratios 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import metrics as m
from foambryo_benchmarks import phantoms as ph
from foambryo_benchmarks.run_case import _json_safe
from foambryo import PressureComputationMethod, TensionComputationMethod, infer_pressures, infer_tensions
from foambryo.dcel import DcelData
from foambryo.validation import mesh_io

DEFAULT_N_RINGS = (8, 16, 32, 64, 128)
DEFAULT_RATIOS = (0.5, 1.0, 1.5)
DEFAULT_FIXED_PHI = 48

#: The two families whose ordering the dataset audit found inverted, plus the force-balance family.
#: Grouping is by *what the method reads off the mesh*, which is the distinction that
#: matters here: measured dihedral angles at triple lines, or discrete energy gradients.
METHOD_FAMILIES = {
    "young_dupre": (
        "YoungDupre",
        "ProjectionYoungDupre",
        "Cotan",
        "InvCotan",
        "Lami",
        "InvLami",
        "LogLami",
    ),
    "variational": ("Variational",),
    "force_balance": ("ForceBalance", "ForceBalanceSoftRow", "ForceBalanceMAP"),
    "per_edge_equilibrium": ("Equilibrium",),
}


def _family_of(method_name: str) -> str:
    """Return the family a method name belongs to, or `"other"`."""
    for family, members in METHOD_FAMILIES.items():
        if method_name in members:
            return family
    return "other"


def measure_doublet(ratio: float, n_rings: int, n_phi: int) -> dict:
    """Run every tension method on one analytic doublet mesh and return its errors.

    Args:
        ratio (float): `gamma_cc / gamma_cm`, in `(0, 2)`.
        n_rings (int): Latitude bands per dome.
        n_phi (int): Azimuthal segments (also the number of triple-line segments).

    Returns:
        dict: `{"n_vertices", "n_triangles", "methods": {name: {...}}}`, each method
            carrying `tension_error_matched_gauge` (median over the 3 interfaces),
            `tension_error_per_key`, `pressure_error_matched_gauge`, `mean_inferred_tension`
            and `wall_time_s`, or `error` if it raised.
    """
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=n_phi)
    truth_tensions = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
    truth_pressures = ph.doublet_ground_truth_pressures(ratio, mean_tension=1.0)
    mesh = DcelData(v, f, regions)

    methods: dict[str, dict] = {}
    for mode in TensionComputationMethod:
        t0 = time.perf_counter()
        try:
            dict_tensions = infer_tensions(mesh, mean_tension=1, mode=mode)
        except Exception as exc:
            methods[mode.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        wall_time = time.perf_counter() - t0

        scale = m.gauge_matching_scale(dict_tensions, truth_tensions)
        matched = m.tension_relative_errors(
            {key: scale * value for key, value in dict_tensions.items()},
            truth_tensions,
        )
        try:
            dict_pressures = infer_pressures(
                mesh,
                dict_tensions,
                mode=PressureComputationMethod.Variational,
                base_pressure=0,
            )
            pressure = m.pressure_relative_errors(
                {key: scale * value for key, value in dict_pressures.items()},
                truth_pressures,
            )["median"]
        except Exception as exc:
            pressure = None
            methods.setdefault(mode.name, {})["pressure_error_note"] = f"{type(exc).__name__}: {exc}"

        methods[mode.name] = {
            "family": _family_of(mode.name),
            "tension_error_matched_gauge": matched["median"],
            "tension_error_max_matched_gauge": float(max(matched["per_key"].values())),
            "tension_error_per_key": matched["per_key"],
            "pressure_error_matched_gauge": pressure,
            "gauge_matching_scale": scale,
            "mean_inferred_tension": float(np.mean(list(dict_tensions.values()))),
            "wall_time_s": wall_time,
        }

    return {
        "n_vertices": len(v),
        "n_triangles": len(f),
        "n_rings": n_rings,
        "n_phi": n_phi,
        # Recorded so the convergence order can be re-expressed in mesh spacing rather than
        # in `n_rings`, which is what makes the extrapolation to a *different* mesh (a
        # `benchmarking-dataset` case) an arithmetic statement instead of an analogy.
        "mean_edge_length": mesh_io.mean_edge_length(v, f),
        "methods": methods,
    }


def fit_order(n_rings: list[int], errors: list[float]) -> dict:
    """Fit `error ~ C * n_rings^-q` by least squares in log-log, with a standard error.

    Args:
        n_rings (list[int]): Resolutions, ascending.
        errors (list[float]): Matching errors; non-finite or non-positive entries are dropped.

    Returns:
        dict: `{"order", "order_stderr", "r_squared", "n_points", "note"}`. `order` is `q`
            (positive = converging). `None` values mean the fit could not be formed.
    """
    x, y = [], []
    for n, e in zip(n_rings, errors, strict=True):
        if e is not None and np.isfinite(e) and e > 0:
            x.append(np.log(n))
            y.append(np.log(e))
    if len(x) < 3:
        return {"order": None, "order_stderr": None, "r_squared": None, "n_points": len(x),
                "note": "fewer than 3 usable points"}
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    design = np.vstack([x_arr, np.ones_like(x_arr)]).T
    coefficients, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    slope, intercept = coefficients
    prediction = design @ coefficients
    residual = y_arr - prediction
    dof = len(x_arr) - 2
    total_variance = float(((y_arr - y_arr.mean()) ** 2).sum())
    # Standard error of the slope: s^2 / sum((x - xbar)^2), the textbook OLS expression.
    s_squared = float((residual**2).sum() / dof) if dof > 0 else float("nan")
    stderr = float(np.sqrt(s_squared / ((x_arr - x_arr.mean()) ** 2).sum())) if dof > 0 else None
    return {
        "order": float(-slope),
        "order_stderr": stderr,
        "r_squared": float(1.0 - float((residual**2).sum()) / total_variance) if total_variance > 0 else None,
        "intercept": float(intercept),
        "n_points": len(x_arr),
        "note": None,
    }


def run_sweep(
    n_rings_list: tuple[int, ...] = DEFAULT_N_RINGS,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    mode: str = "uniform",
    fixed_phi: int = DEFAULT_FIXED_PHI,
    fit_last: int | None = None,
) -> dict:
    """Sweep resolutions and ratios, then fit each method's convergence order.

    Args:
        n_rings_list (tuple[int, ...], optional): Resolutions to run, ascending.
        ratios (tuple[float, ...], optional): `gamma_cc / gamma_cm` values.
        mode (str, optional): `"uniform"` (`n_phi = 2 * n_rings`) or `"fixed_phi"`.
        fixed_phi (int, optional): `n_phi` in `"fixed_phi"` mode.
        fit_last (int | None, optional): Fit the order on the last this-many resolutions
            only (the asymptotic regime). Defaults to all of them.

    Returns:
        dict: `{"mode", "n_rings", "cases": {ratio: {...}}, "orders": {ratio: {method: {...}}}}`.
    """
    cases: dict[str, dict] = {}
    orders: dict[str, dict] = {}
    for ratio in ratios:
        per_resolution = {}
        for n_rings in n_rings_list:
            n_phi = 2 * n_rings if mode == "uniform" else fixed_phi
            per_resolution[str(n_rings)] = measure_doublet(ratio, n_rings, n_phi)
        cases[str(ratio)] = per_resolution

        method_names = sorted({name for record in per_resolution.values() for name in record["methods"]})
        fitted = {}
        for name in method_names:
            errors = [per_resolution[str(n)]["methods"].get(name, {}).get("tension_error_matched_gauge")
                      for n in n_rings_list]
            keep = slice(-fit_last, None) if fit_last else slice(None)
            fitted[name] = {
                "family": _family_of(name),
                "errors": errors,
                **fit_order(list(n_rings_list)[keep], errors[keep]),
                # Also fit the three finest resolutions on their own. The coarsest points are
                # routinely outside the asymptotic regime (several methods here are
                # non-monotone between n_rings 8 and 16), and a single fit over all five
                # points then reports an order that is really an average over two regimes.
                # Reporting both is what keeps a poor fit visible as a poor fit.
                "asymptotic": fit_order(list(n_rings_list)[-3:], errors[-3:]),
            }
        orders[str(ratio)] = fitted
    return {
        "mode": mode,
        "fixed_phi": fixed_phi if mode == "fixed_phi" else None,
        "n_rings": list(n_rings_list),
        "fit_last": fit_last,
        "cases": cases,
        "orders": orders,
    }


def markdown_table(sweep: dict) -> str:
    """Render one sweep as a markdown error table plus a fitted-order table per ratio."""
    def order_text(entry: dict) -> str:
        order, stderr = entry["order"], entry["order_stderr"]
        if order is None:
            return "—"
        return f"{order:+.2f} ± {stderr:.2f}" if stderr else f"{order:+.2f}"

    lines: list[str] = []
    n_rings = sweep["n_rings"]
    for ratio, fitted in sweep["orders"].items():
        spacings = [sweep["cases"][ratio][str(n)]["mean_edge_length"] for n in n_rings]
        lines.append(f"### ratio = {ratio}, mode = `{sweep['mode']}`\n")
        lines.append("mean edge length: " + ", ".join(f"{h:.4f}" for h in spacings) + "\n")
        header = " | ".join(f"n_rings={n}" for n in n_rings)
        lines.append(f"| method | family | {header} | `q` (all) | R² | `q` (finest 3) |")
        lines.append("|---|---|" + "---|" * (len(n_rings) + 3))
        for name, entry in sorted(fitted.items(), key=lambda kv: (kv[1]["family"], kv[0])):
            cells = " | ".join("—" if e is None else f"{e:.2e}" for e in entry["errors"])
            r2 = entry["r_squared"]
            lines.append(
                f"| `{name}` | {entry['family']} | {cells} | {order_text(entry)} | "
                f"{'—' if r2 is None else f'{r2:.3f}'} | {order_text(entry['asymptotic'])} |",
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the sweep from the command line and write JSON + markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rings", type=int, nargs="+", default=list(DEFAULT_N_RINGS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--modes", type=str, nargs="+", default=["uniform", "fixed_phi"],
                        choices=["uniform", "fixed_phi"])
    parser.add_argument("--fixed-phi", type=int, default=DEFAULT_FIXED_PHI)
    parser.add_argument("--fit-last", type=int, default=None,
                        help="Fit the order on the last N resolutions only (asymptotic regime)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "baseline")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweeps = {}
    for mode in args.modes:
        sweep = run_sweep(
            tuple(args.n_rings), tuple(args.ratios), mode=mode,
            fixed_phi=args.fixed_phi, fit_last=args.fit_last,
        )
        sweeps[mode] = sweep
        print(markdown_table(sweep))

    (args.output_dir / "doublet_convergence.json").write_text(
        json.dumps(_json_safe(sweeps), indent=2, sort_keys=True) + "\n",
    )
    (args.output_dir / "doublet_convergence.md").write_text(
        "# Doublet mesh-convergence sweep (the force-balance work)\n\n"
        "Gauge-matched median relative tension error over the 3 interfaces, and the fitted\n"
        "convergence order `q` in `error ~ n_rings^-q`. Generated by\n"
        "`python -m foambryo_benchmarks.run_doublet_convergence`; see that module's docstring for\n"
        "the two resolution knobs and why both modes are reported.\n\n"
        + "\n".join(markdown_table(s) for s in sweeps.values()),
    )


if __name__ == "__main__":
    main()
