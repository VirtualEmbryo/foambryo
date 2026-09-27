"""F4 — the angle-error decomposition: C*eps/ell calibration, measured points, line/apex terms.

The junction-angle error budget: an analytic 120deg
trijunction perturbed by i.i.d. vertex noise gives a response linear in eps/ell,
`C = 33.2 +/- 1.2 deg` per unit eps/ell at ell=8.3 (close to the real mesh's ell_median). The
model describes ~90% of already-measured error with no fitted parameter, decomposes into a
line term and an apex term that explain junction protection's apparent null result, and is
anti-predictive for forecasting an intervention (it over-predicted offset exclusion's effect
by 3x). This figure shows the calibration line with its measured points, the line/apex terms
by variant, and the ground-truth 0.34deg floor.

Data, all real and already committed by the junction-protection and offset-exclusion work
(no dw3d re-run needed):
- `delaunay-watershed-3d/dw3d_benchmarks/baseline/angle_error_budget_default_offset_excluded_link_checked.json`:
  `wedge` (the analytic calibration curve + fitted C per mode, at ell=8.3/16.0/4.0),
  `effective_vertex_error` (per-case, per-variant eps_line/eps_apex against the registered
  ground-truth surface), `ground_truth_ladder` (the ground-truth floor).
- Measured total junction-angle error per variant: aggregated live from each variant's own
  committed per-case records (`ground_truth.angle_error_vs_neumann`), the same convention
  the `dw3d` benchmark ledger uses (median of per-case medians).

Run with: `uv run python -m foambryo_benchmarks.figures.f4_angle_error_decomposition` from `foambryo/`.
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict

import numpy as np

from foambryo_benchmarks.figures._style import (
    AQUA,
    BLUE,
    DW3D_BASELINE,
    ORANGE,
    RED,
    SINGLE_COL_IN,
    TEXT_SECONDARY,
    apply_style,
    save_fig,
    strip_spines,
)

VARIANT_DIRS = {
    "deterministic": "deterministic",
    "offset included": "junction_protected",
    "offset excluded": "offset_excluded",
    "link checked (default)": "offset_excluded_linkcheck",
}
EFFECTIVE_ERROR_VARIANT_KEY = {
    "offset included": "junction_protected",
    "offset excluded": "offset_excluded",
    "link checked (default)": "default",  # this baseline's own label for the current default
}


def measured_angle_error_by_variant() -> tuple[dict[str, float], float]:
    medians = {}
    gt_medians = []
    for label, dirname in VARIANT_DIRS.items():
        case_medians = []
        for f in sorted(glob.glob(str(DW3D_BASELINE / dirname / "dataset_*_md3.json"))):
            rec = json.loads(open(f).read())
            gt = rec.get("ground_truth")
            if gt and gt.get("angle_error_vs_neumann"):
                case_medians.append(gt["angle_error_vs_neumann"]["angle_error_median_deg"])
                gt_medians.append(gt["reference_mesh_angle_error_vs_neumann"]["angle_error_median_deg"])
        if not case_medians:
            raise RuntimeError(f"no ground-truth angle data for variant '{label}' ({dirname})")
        medians[label] = statistics.median(case_medians)
    gt_floor = statistics.median(gt_medians)
    return medians, gt_floor


def load_calibration() -> dict:
    return json.loads(
        (DW3D_BASELINE / "angle_error_budget_default_offset_excluded_link_checked.json").read_text(),
    )


# epsilon_of_the_estimator_stencil's own `variant` labels predate the link-checked default
# flip: "default" means offset_included (boundary layer + junction protection),
# "offset_excluded"/"offset_excluded_linkcheck" mean the weld/link-checked exclusion --
# confirmed against the recorded model_prediction_deg values
# (5.06/4.14/4.25).
STENCIL_VARIANT_LABEL = {
    "default": "offset included",
    "offset_excluded": "offset excluded",
    "offset_excluded_linkcheck": "link checked (default)",
}
STENCIL_VARIANT_DIR = {
    "default": "junction_protected",
    "offset_excluded": "offset_excluded",
    "offset_excluded_linkcheck": "offset_excluded_linkcheck",
}


def measured_angle_for_case(dirname: str, case: str) -> float:
    rec = json.loads((DW3D_BASELINE / dirname / f"dataset_{case}_labels_filled_md3.json").read_text())
    return rec["ground_truth"]["angle_error_vs_neumann"]["angle_error_median_deg"]


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    budget = load_calibration()
    wedge = budget["wedge"]["8.3"]  # ell close to the real mesh's ell_median
    measured, gt_floor = measured_angle_error_by_variant()

    fig, axes = plt.subplots(1, 3, figsize=(3 * SINGLE_COL_IN * 0.92, SINGLE_COL_IN * 0.95))

    # --- Panel (a): the analytic calibration curve, all three variants (all/line/apex) -----
    ax = axes[0]
    mode_style = {
        "all": (TEXT_SECONDARY, "all vertices"),
        "line_only": (BLUE, "line vertices"),
        "apex_only": (ORANGE, "apex vertices"),
    }
    for mode, (color, label) in mode_style.items():
        curve = wedge["modes"][mode]["curve"]
        eps = np.array([c["epsilon"] for c in curve])
        mean_deg = np.array([c["mean_deg"] for c in curve])
        sd_deg = np.array([c["sd_deg"] for c in curve])
        ax.errorbar(eps, mean_deg, yerr=sd_deg, fmt="o-", color=color, markersize=3,
                    linewidth=1.2, capsize=2, label=f"{label} (C={wedge['modes'][mode]['C_deg_per_ratio']:.1f}°)")

    # Overlay the real measured points: stencil-epsilon/ell vs. the case's own measured
    # junction-angle error, per (case, variant) -- this is what the model is judged against,
    # and it is why the model is a good description (close to the
    # "all vertices" curve) yet a bad forecast (the population-epsilon curve, not plotted
    # here, would sit far above these points).
    stencil_records = budget["epsilon_of_the_estimator_stencil"]
    measured_x, measured_y = [], []
    for r in stencil_records:
        eps_ratio = r["epsilon_stencil_rms"] / r["ell_median_triple_line_edge"]
        angle = measured_angle_for_case(STENCIL_VARIANT_DIR[r["variant"]], r["case"])
        measured_x.append(eps_ratio)
        measured_y.append(angle)
    ax.scatter(measured_x, measured_y, marker="^", s=26, facecolor=RED, edgecolor="white",
               linewidth=0.4, zorder=5, label="measured (case, variant) pairs")

    ax.set_xlabel(r"vertex noise ratio $\epsilon/\ell$")
    ax.set_ylabel("angle error (deg)")
    ax.set_title(f"(a) calibration at $\\ell$={wedge['edge_length']:.1f}, with real points", fontsize=8.2)
    ax.legend(loc="upper left", fontsize=6.0)
    strip_spines(ax)

    # --- Panel (b): line term vs. apex term epsilon, by variant -----------------------------
    # This is the decomposition that explains junction protection's apparent null result:
    # junction protection halves the line term while the boundary
    # layer introduces an apex term of almost exactly the same size, so only the *total*
    # looked unchanged.
    axb = axes[1]
    eve_records = budget["effective_vertex_error"]
    by_variant_line: dict[str, list[float]] = defaultdict(list)
    by_variant_apex: dict[str, list[float]] = defaultdict(list)
    for r in eve_records:
        label = STENCIL_VARIANT_LABEL[r["variant"]]
        by_variant_line[label].append(r["epsilon_line_median"])
        by_variant_apex[label].append(r["epsilon_apex_median"])
    variant_labels_b = ["offset included", "offset excluded", "link checked (default)"]
    x = np.arange(len(variant_labels_b))
    width = 0.36
    line_medians = [statistics.median(by_variant_line[v]) for v in variant_labels_b]
    apex_medians = [statistics.median(by_variant_apex[v]) for v in variant_labels_b]
    axb.bar(x - width / 2, line_medians, width=width, color=BLUE, label=r"$\epsilon_{\rm line}$")
    axb.bar(x + width / 2, apex_medians, width=width, color=ORANGE, label=r"$\epsilon_{\rm apex}$")
    axb.set_xticks(x)
    axb.set_xticklabels(variant_labels_b, fontsize=6.8)
    axb.set_ylabel("median vertex error $\\epsilon$ (voxels)")
    axb.set_title("(b) line vs. apex term, by variant", fontsize=8.2)
    axb.legend(loc="upper right", fontsize=6.4)
    strip_spines(axb)

    # --- Panel (c): measured total angle error per variant, with the ground-truth floor -----
    ax2 = axes[2]
    labels = list(measured.keys())
    values = [measured[k] for k in labels]
    colors = [AQUA, RED, "#c98500", BLUE]
    bars = ax2.bar(range(len(labels)), values, color=colors[: len(labels)], width=0.6,
                    edgecolor="white", linewidth=0.5)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.3, f"{v:.2f}°", ha="center", fontsize=6.4, color="#33322f")
    ax2.axhline(gt_floor, color="#8a8a86", linewidth=1.1, linestyle="--", zorder=1)
    ax2.text(len(labels) - 0.4, gt_floor + 0.4, f"ground truth: {gt_floor:.2f}°",
             fontsize=6.4, ha="right", color="#52514e", style="italic")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=6.8, rotation=12)
    ax2.set_ylabel("median junction-angle error vs. Neumann (deg)")
    ax2.set_title("(c) measured, 47 cases", fontsize=8.2)
    strip_spines(ax2)

    fig.suptitle(
        "Junction-angle error: a calibrated description, an unforecastable intervention",
        fontsize=9.2, y=1.04,
    )
    fig.tight_layout()
    save_fig(fig, "F4_angle_error_decomposition")


if __name__ == "__main__":
    main()
