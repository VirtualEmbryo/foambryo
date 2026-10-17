"""Shared style, palette and I/O helpers for the figure set.

Every figure script in this package imports from here so the whole set reads as one
system: same fonts, same colour ramp, same save/verify discipline. The palette is the
validated colourblind-safe categorical/sequential/diverging set from the project's
`dataviz` skill reference (`references/palette.md`) — a fixed hue order, not re-derived
per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------------
# This file lives at <workspace_root>/foambryo/foambryo_benchmarks/figures/_style.py
FIGURES_SCRIPT_DIR = Path(__file__).resolve().parent
FOAMBRYO_REPO = FIGURES_SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = FOAMBRYO_REPO.parent
DW3D_REPO = WORKSPACE_ROOT / "delaunay-watershed-3d"
DW3D_VARIANT_COMPARISON_OUTPUTS = FOAMBRYO_REPO / "foambryo_benchmarks" / "outputs" / "dw3d_variant_comparison"
CONVERGED_MESHES = WORKSPACE_ROOT / "converged-meshes"
# Git-ignored (`foambryo_benchmarks/outputs/` in .gitignore). Created by `save_fig`, never on
# import: importing this module must not write anywhere, least of all into the workspace root.
FIGURES_OUT = FOAMBRYO_REPO / "foambryo_benchmarks" / "outputs" / "figures"

DW3D_BASELINE = DW3D_REPO / "dw3d_benchmarks" / "baseline"
FOAMBRYO_BASELINE = FOAMBRYO_REPO / "foambryo_benchmarks" / "baseline"

# --- Palette (dataviz skill reference palette, fixed categorical order) ------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Two-series comparisons throughout this figure set use the same fixed pair:
# ForceBalanceKKT/Variational (operator-inverting family) vs YoungDupre (angle family).
COLOR_FORCEBALANCE = BLUE
COLOR_YOUNGDUPRE = ORANGE

# Sequential single-hue ramp (blue, light -> dark), for magnitude/ordinal encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

# Diverging pair (blue <-> red), neutral grey midpoint.
DIVERGING = (BLUE, "#f0efec", RED)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8a86"
GRID = "#e3e2dd"
SURFACE = "#fcfcfb"

SINGLE_COL_IN = 3.5  # inches, single-column journal width


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(top=False, right=False)


def save_fig(fig, name: str) -> tuple[Path, Path]:
    """Save `fig` as both PDF (vector, for LaTeX) and PNG, then verify both are real.

    Returns (pdf_path, png_path). Raises if either file is empty or fails to re-open —
    by design, a script must not claim success for
    an output it never actually wrote.
    """
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_OUT / f"{name}.pdf"
    png_path = FIGURES_OUT / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    for path in (pdf_path, png_path):
        size = path.stat().st_size
        if size == 0:
            raise RuntimeError(f"{path} is zero bytes")

    with open(pdf_path, "rb") as fh:
        header = fh.read(5)
        if header != b"%PDF-":
            raise RuntimeError(f"{pdf_path} does not start with a PDF header")

    from PIL import Image

    with Image.open(png_path) as im:
        im.verify()
    with Image.open(png_path) as im:
        arr_extrema = im.convert("L").getextrema()
        if arr_extrema[0] == arr_extrema[1]:
            raise RuntimeError(f"{png_path} is a flat/blank image")

    print(f"[figures] wrote {pdf_path.name} ({pdf_path.stat().st_size} B) and "
          f"{png_path.name} ({png_path.stat().st_size} B), both verified re-openable")
    return pdf_path, png_path
