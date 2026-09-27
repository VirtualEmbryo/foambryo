"""F2 -- the point budget and the boundary layer.

The root cause found for `dw3d`'s mesh defects: the surface:interior point ratio is imbalanced
(44:1 on the 6-cell image `3.tif` at `min_distance = 3`: 5 911 interface points against 133
interior points), which starves the watershed and produces all-surface
tetrahedra with degenerate scores. This figure tracks surface vs. interior sample counts and
the all-surface-tetrahedron fraction across the four configurations that matter for that
story: `dithered` (the original), `deterministic` (no boundary layer yet), `offset_included`
(the boundary layer + junction protection), and the current default (link checked, same
point placement as offset_included, only the extracted surface differs).

Data: `delaunay-watershed-3d/dw3d_benchmarks/baseline/<variant>/dataset_*_md3.json`, all 47/51
cases, `point_placement` (`n_corner_points`, `n_interface_points`, `n_interior_points`) and
`tetrahedron_surface_stats` (`fraction_all_surface_tets`) -- committed golden-master records,
no dw3d re-run needed.

Run with: `uv run python -m foambryo_benchmarks.figures.f2_point_budget_boundary_layer` from `foambryo/`.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

from foambryo_benchmarks.figures._style import (
    AQUA,
    BLUE,
    DW3D_BASELINE,
    ORANGE,
    RED,
    SINGLE_COL_IN,
    apply_style,
    save_fig,
    strip_spines,
)

VARIANT_ORDER = [
    ("dithered", "dithered"),
    ("deterministic", "deterministic"),
    ("junction_protected", "offset included"),
    ("offset_excluded_linkcheck", "link checked\n(current default)"),
]
COLORS = [AQUA, ORANGE, RED, BLUE]


def load_variant(dirname: str) -> dict:
    interior = []
    surface = []
    all_surface_frac = []
    for f in sorted(glob.glob(str(DW3D_BASELINE / dirname / "dataset_*_md3.json"))):
        rec = json.loads(open(f).read())
        pp = rec["point_placement"]
        n_surface = pp["n_interface_points"] + pp.get("n_corner_points", 0)
        interior.append(pp["n_interior_points"])
        surface.append(n_surface)
        all_surface_frac.append(rec["tetrahedron_surface_stats"]["fraction_all_surface_tets"])
    return {
        "interior": np.array(interior, dtype=float),
        "surface": np.array(surface, dtype=float),
        "all_surface_frac": np.array(all_surface_frac, dtype=float) * 100,
    }


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    data = {dirname: load_variant(dirname) for dirname, _ in VARIANT_ORDER}
    for dirname, d in data.items():
        if len(d["interior"]) == 0:
            raise RuntimeError(f"no records found for variant directory '{dirname}'")

    fig, axes = plt.subplots(1, 2, figsize=(2 * SINGLE_COL_IN, SINGLE_COL_IN * 0.95))

    # --- Panel (a): median surface:interior ratio, log scale, with the case spread ---------
    ax = axes[0]
    positions = np.arange(len(VARIANT_ORDER))
    ratios = [data[d]["surface"] / np.maximum(data[d]["interior"], 1) for d, _ in VARIANT_ORDER]
    bp = ax.boxplot(ratios, positions=positions, widths=0.55, whis=(5, 95), patch_artist=True,
                     showfliers=True,
                     flierprops={"marker": "o", "markersize": 1.8, "markerfacecolor": "#52514e",
                                 "markeredgecolor": "none", "alpha": 0.4},
                     medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in VARIANT_ORDER], fontsize=6.8)
    ax.set_ylabel("surface : interior point ratio\n(per case, log scale)")
    ax.set_title("(a) point budget, 47-51 cases", fontsize=8.4)
    strip_spines(ax)

    # --- Panel (b): all-surface-tetrahedron fraction ---------------------------------------
    ax2 = axes[1]
    fracs = [data[d]["all_surface_frac"] for d, _ in VARIANT_ORDER]
    bp2 = ax2.boxplot(fracs, positions=positions, widths=0.55, whis=(5, 95), patch_artist=True,
                       showfliers=True,
                       flierprops={"marker": "o", "markersize": 1.8, "markerfacecolor": "#52514e",
                                   "markeredgecolor": "none", "alpha": 0.4},
                       medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp2["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([label for _, label in VARIANT_ORDER], fontsize=6.8)
    ax2.set_ylabel("all-surface tetrahedra (% of mesh)")
    ax2.set_title("(b) all-surface-tet fraction", fontsize=8.4)
    strip_spines(ax2)

    fig.suptitle("The point budget and the boundary layer", fontsize=9.4, y=1.03)
    fig.tight_layout()
    save_fig(fig, "F2_point_budget_boundary_layer")


if __name__ == "__main__":
    main()
