"""F5 — error vs rho_free, per population, per method.

The figure for the `rho_free` mechanism: `ForceBalanceKKT`'s tension
error tracks the mesh's own distance from equilibrium (`rho_free`) within a population, but
pooling the 42-case ground-truth dataset with the 72-mesh converged cohort produces a pooled
regression whose slope is not the within-population one (the pre-registered pooled test gives
`ForceBalanceKKT` a Spearman r of 0.116 over the 114 pooled meshes, against 0.894 within the
72-mesh cohort alone and 0.066 within the 42). The leading candidate explanation, not confirmed
(it has not been tested directly, for example by a regression with a per-population intercept), is
that the two populations sit at different intercepts -- a Simpson's-paradox-style aggregation.
Junction-graph fragmentation was tested and ruled out as the explanation.

Data, both real, no recomputation of the correlation itself:
- 42 equilibrium ground-truth cases: `foambryo_benchmarks/baseline/attribution.json`
  (`rho_free_p1`, `e_inference_detail.gauge_matched.median`, deduplicated across `dw3d_variant`
  since both are P1-only/variant-independent quantities).
- 72 converged-cohort meshes: `foambryo_benchmarks/baseline/converged_cohort.json`
  (`rho_free`, `methods.<method>.tension_relative_error_matched_gauge.median`) — the
  baseline after the junction-graph fragmentation fix.

Run with: `uv run python -m foambryo_benchmarks.figures.f5_error_vs_rho_free` from `foambryo/`.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from foambryo_benchmarks.figures._style import (
    COLOR_FORCEBALANCE,
    COLOR_YOUNGDUPRE,
    FOAMBRYO_BASELINE,
    SINGLE_COL_IN,
    apply_style,
    save_fig,
    strip_spines,
)

METHODS = ["ForceBalanceKKT", "YoungDupre"]
METHOD_COLOR = {"ForceBalanceKKT": COLOR_FORCEBALANCE, "YoungDupre": COLOR_YOUNGDUPRE}
POP_47_COLOR = "#52514e"  # neutral dark grey -- the smaller, lower-rho_free population
POP_72_MARKER = "s"
POP_47_MARKER = "o"


def load_47_population() -> dict[str, list[tuple[float, float]]]:
    """(rho_free, e_inference) pairs per method, deduplicated across dw3d_variant."""
    records = json.loads((FOAMBRYO_BASELINE / "attribution.json").read_text())
    seen = {}
    for r in records:
        method = r["method"]
        if method not in METHODS:
            continue
        key = (method, r["case"])
        # rho_free_p1 / e_inference are P1-only (ground-truth-mesh) quantities and must be
        # identical across dw3d_variant for a given (case, method) -- verified below.
        val = (r["rho_free_p1"], r["e_inference_detail"]["gauge_matched"]["median"])
        if key in seen:
            prev = seen[key]
            if not np.isclose(prev[0], val[0]) or not np.isclose(prev[1], val[1]):
                raise RuntimeError(f"{key} disagrees across dw3d_variant: {prev} vs {val}")
        else:
            seen[key] = val
    out: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}
    for (method, _case), val in seen.items():
        out[method].append(val)
    return out


def load_72_population() -> dict[str, list[tuple[float, float]]]:
    records = json.loads((FOAMBRYO_BASELINE / "converged_cohort.json").read_text())
    out: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}
    for r in records:
        rho_free = r["rho_free"]
        for method in METHODS:
            median_err = r["methods"][method]["tension_relative_error_matched_gauge"]["median"]
            out[method].append((rho_free, median_err))
    return out


def fit_line_log_y(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit of log10(y) vs log10(x); returns (x_grid, y_grid) for plotting."""
    mask = (xs > 0) & (ys > 0)
    lx, ly = np.log10(xs[mask]), np.log10(ys[mask])
    slope, intercept, *_ = stats.linregress(lx, ly)
    x_grid = np.logspace(np.log10(xs[mask].min()), np.log10(xs[mask].max()), 50)
    y_grid = 10 ** (intercept + slope * np.log10(x_grid))
    return x_grid, y_grid


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    pop47 = load_47_population()
    pop72 = load_72_population()

    fig, axes = plt.subplots(1, 2, figsize=(2 * SINGLE_COL_IN, SINGLE_COL_IN * 0.95), sharex=False)

    for ax, method in zip(axes, METHODS):
        x47 = np.array([p[0] for p in pop47[method]])
        y47 = np.array([p[1] for p in pop47[method]])
        x72 = np.array([p[0] for p in pop72[method]])
        y72 = np.array([p[1] for p in pop72[method]])

        color = METHOD_COLOR[method]

        ax.scatter(x47, y47, s=10, marker=POP_47_MARKER, facecolor="none",
                   edgecolor=POP_47_COLOR, linewidth=0.8, label="42 ground-truth cases", zorder=3)
        ax.scatter(x72, y72, s=10, marker=POP_72_MARKER, facecolor=color, alpha=0.75,
                   edgecolor="none", label="72 converged-cohort meshes", zorder=3)

        # Within-population regression lines (solid).
        xg, yg = fit_line_log_y(x47, y47)
        ax.plot(xg, yg, color=POP_47_COLOR, linewidth=1.4, zorder=2)
        xg, yg = fit_line_log_y(x72, y72)
        ax.plot(xg, yg, color=color, linewidth=1.4, zorder=2)

        # Pooled regression (dashed), to make the intercept mismatch visible rather than
        # asserted -- this is the whole point of the figure.
        x_all = np.concatenate([x47, x72])
        y_all = np.concatenate([y47, y72])
        xg, yg = fit_line_log_y(x_all, y_all)
        ax.plot(xg, yg, color="black", linewidth=1.1, linestyle=(0, (4, 2)), zorder=1,
                label="pooled fit (mis-specified)")

        rho_pooled, p_pooled = stats.spearmanr(x_all, y_all)
        rho_47, _ = stats.spearmanr(x47, y47)
        rho_72, _ = stats.spearmanr(x72, y72)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho_{\mathrm{free}}$")
        ax.set_ylabel("gauge-matched median\ntension error" if method == METHODS[0] else "")
        ax.set_title(method, fontsize=9)
        strip_spines(ax)

        ax.text(
            0.03, 0.97,
            f"pooled $r_S$={rho_pooled:.2f}\n42-only $r_S$={rho_47:.2f}\n72-only $r_S$={rho_72:.2f}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left", color="#33322f",
        )

    axes[0].legend(loc="lower right", fontsize=6, markerscale=1.2)
    fig.suptitle(
        "Error vs. mesh distance from equilibrium ($\\rho_{\\mathrm{free}}$), by population",
        fontsize=9.5, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "F5_error_vs_rho_free")


if __name__ == "__main__":
    main()
