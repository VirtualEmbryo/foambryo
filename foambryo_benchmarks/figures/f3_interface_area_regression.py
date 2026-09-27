"""F3 — the interface-area regression, signed error by variant, all six configurations.

`dw3d`'s default has shipped six distinct point-placement/extraction configurations; the
boundary layer causes a +31.8% interface-area over-estimate, junction protection halves
it, and offset exclusion (weld or link-checked) removes it by excluding the boundary-layer
offsets from the extracted surface (signed median error over the 47 ground-truth cases at
`min_distance = 3`). This figure makes the boundary-layer -> junction-protection
-> link-checked attribution chain visible directly from the per-case, per-interface data.

Data: `delaunay-watershed-3d/dw3d_benchmarks/baseline/interface_area_error_default_offset_excluded_link_checked.json` -- the
already-committed ledger covering all 47 ground-truth cases x 6 variants (282 records), each
carrying `signed_relative_error_median` (the case-level summary) and
`per_interface_signed_relative_error` (per label-pair). Note this file's own `variant` field
labels the offset-included configuration "default" (it predates the link-checked flip); relabelled here to
match this package's naming (`offset_included`) so the figure is not misread.

Run with: `uv run python -m foambryo_benchmarks.figures.f3_interface_area_regression` from `foambryo/`.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from foambryo_benchmarks.figures._style import (
    BLUE,
    DW3D_BASELINE,
    RED,
    SINGLE_COL_IN,
    TEXT_SECONDARY,
    apply_style,
    save_fig,
    strip_spines,
)

# Attribution-chain order, oldest to newest, with the raw file's own `variant` field mapped
# to the report's naming (the file predates the link-checked default flip, so its own
# "default" label means the offset-included configuration, not the current default).
VARIANT_ORDER = [
    # First element is the literal `variant` value recorded in the committed JSON (never
    # renamed, so it keeps matching historical data); third is the chart's display label.
    ("v0_3", "dithered", "dithered"),
    ("a1b", "deterministic", "deterministic"),
    ("boundary_layer", "boundary_layer", "boundary layer alone"),
    ("default", "offset_included", "boundary layer + junction protection"),
    ("offset_excluded", "offset_excluded", "offset excluded (weld)"),
    ("offset_excluded_linkcheck", "offset_excluded_linkcheck", "link checked"),
]


def load_pooled_errors() -> dict[str, np.ndarray]:
    """Every interface's signed relative area error, pooled over all 47 cases, per variant."""
    records = json.loads(
        (DW3D_BASELINE / "interface_area_error_default_offset_excluded_link_checked.json").read_text(),
    )
    by_raw_variant: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_raw_variant[r["variant"]].extend(r["per_interface_signed_relative_error"].values())
    return {raw: np.array(vals) * 100 for raw, vals in by_raw_variant.items()}


def load_case_medians() -> dict[str, np.ndarray]:
    records = json.loads(
        (DW3D_BASELINE / "interface_area_error_default_offset_excluded_link_checked.json").read_text(),
    )
    by_raw_variant: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_raw_variant[r["variant"]].append(r["signed_relative_error_median"] * 100)
    return {raw: np.array(vals) for raw, vals in by_raw_variant.items()}


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    pooled = load_pooled_errors()
    case_medians = load_case_medians()

    for raw, _, _ in VARIANT_ORDER:
        if raw not in pooled:
            raise RuntimeError(f"variant '{raw}' missing from committed ledger")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 1.05))

    positions = np.arange(len(VARIANT_ORDER))
    data = [pooled[raw] for raw, _, _ in VARIANT_ORDER]

    # Boxplot on the pooled per-interface distribution: median + IQR box, whiskers at the
    # 5th/95th percentile (close to the report's own p90 convention), outliers as small dots
    # -- a boxplot reads the heavy right tail (boundary_layer alone reaches p90 ~66%) far
    # more legibly than a KDE-based violin would.
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, whis=(5, 95), patch_artist=True,
        showfliers=True, flierprops={"marker": "o", "markersize": 1.6, "markerfacecolor": TEXT_SECONDARY,
                                      "markeredgecolor": "none", "alpha": 0.35},
        medianprops={"color": RED, "linewidth": 1.6},
        boxprops={"facecolor": BLUE, "alpha": 0.35, "edgecolor": "#1c5cab"},
        whiskerprops={"color": "#1c5cab"},
        capprops={"color": "#1c5cab"},
    )

    # Per-case medians overlaid as a jittered strip -- the quantity the signed-median table
    # is built from.
    rng = np.random.default_rng(0)
    for i, (raw, _, _) in enumerate(VARIANT_ORDER):
        jitter = rng.uniform(-0.16, 0.16, size=len(case_medians[raw]))
        ax.scatter(
            np.full_like(case_medians[raw], i) + jitter, case_medians[raw],
            s=6, color="#33322f", alpha=0.6, zorder=3, linewidth=0,
        )

    ax.axhline(0, color="#8a8a86", linewidth=0.8, linestyle="--", zorder=1)

    # Shade the two configurations that carry the regression (boundary layer alone,
    # boundary layer + junction protection), rather than cluttering the plot with arrows --
    # the attribution chain (the boundary layer causes it, junction protection halves it,
    # offset exclusion removes it) is stated in the caption/README, not redrawn on the axes.
    ax.axvspan(1.5, 3.5, color=RED, alpha=0.07, zorder=0)
    ax.text(2.5, 68, "regression window\n(boundary layer,\nboundary layer + junction protection)",
            fontsize=6.4, ha="center", color="#8a3a3a", style="italic")

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, _, label in VARIANT_ORDER], fontsize=7.2)
    ax.set_ylabel("signed relative interface-area error (%)")
    ax.set_title("Interface-area regression, dw3d's six configurations", fontsize=8.6)
    ax.set_ylim(-15, 92)
    strip_spines(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#33322f",
                   markersize=4, alpha=0.7, label="per-case median (47 cases)"),
        plt.Line2D([0], [0], color=RED, linewidth=1.6, label="pooled median (all interfaces)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.2)

    fig.tight_layout()
    save_fig(fig, "F3_interface_area_regression")


if __name__ == "__main__":
    main()
